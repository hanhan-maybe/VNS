# V5 通用个体化 MCU 系统实现状态

## 1. 当前结论

V5 的 Pressure-only Python→C 算法移植、F37/F26 双动物回放以及通用
SubjectConfig 动态切换已经通过 PC host 验证。系统尚未完成 STM32N657
板上 HIL，因此当前状态是：

`HOST_VALIDATED_STM32_HIL_REQUIRED`

物理刺激保持关闭：`shadow_mode=true`、`stimulation_enabled=false`，并且
`V5_ALLOW_PHYSICAL_STIMULATION=0` 是默认编译配置。

## 2. 系统架构

```text
100 Hz bladder pressure + validity
  -> v5_candidate (25 s baseline + adaptive CLEAR history)
  -> v5_features (15-D frozen P-EARLY)
  -> v5_model (individual scaler + logistic regression)
  -> v5_runtime (T0-only, one trigger per candidate event)
  -> v5_stim_fsm (IDLE/ACTIVE/REFRACTORY/FAULT)
```

Python 负责校准、冻结模型、导出 SubjectConfig 和生成 Golden Reference；
MCU 只接收压力、有效性和已冻结配置，不接收 teacher label、尿液、Volume、
未来事件或 subject ID 分类特征。

## 3. F37/F26 参数来源

- STxF37：B01–B04 calibration，B05–B07 frozen test；候选 prior 使用冻结
  338 参数表的群体中位数。
- STxF26：B01–B13 calibration，B14–B16 frozen test；候选 prior 使用该动物
  在冻结参数表中的 row。
- 两只动物不混合训练。相同 C Runtime 仅更换 244-byte SubjectConfig。

## 4. SubjectConfig

二进制格式包含 magic、version、feature count、15维 feature-order CRC32、
model SHA-256、scaler mean/scale、LR coefficients/intercept、个体阈值、候选
prior 和整体 CRC32。错误版本、特征 schema、模型参数或 CRC 会拒绝加载。

生成文件位于 `data/NVC_V5/mcu_config/`（数据目录不提交 Git）：

- `STxF37_v5_subject_config.json/.bin`
- `STxF26_v5_subject_config.json/.bin`

## 5. Gate 状态

| Gate | 状态 | 主要结果 |
|---|---|---|
| 1 Model parity | PASS | F37 max `5.96e-08`; F26 max `1.19e-07`; 0 classification mismatch |
| 2 T0-only Runtime | PASS | T0 控制 virtual trigger；T1 仅 debug；默认 shadow |
| 3 Runtime parity | PASS | 3,390 registered updates；score/state/trigger/event mismatch=0 |
| 4 Candidate C | IMPLEMENTED | 移植 frozen adaptive detector；CLEAR history 跨 cycle |
| 5 Candidate parity | PASS | F37 9/9、F26 6/6；起点一致；F26 两个终点差 1 sample |
| 6 P-EARLY C | IMPLEMENTED | 固定 15维顺序；portable detrend/Hann/DFT/PSD |
| 7 Feature parity | PASS | 2,742 rows；15维全部 PASS，无系统偏差 |
| 8 Full C replay | PASS | score max `2.98e-07`; F37 trigger 9/9、F26 6/6；T0一致 |
| 9 SubjectConfig | PASS | JSON/header/binary、schema hash、model hash、CRC |
| 10 Config swap | PASS | 同一 executable 先载 F37、再载 F26，完整回放均 PASS |
| 11 STM32 HIL | NOT EXECUTED | 需要 NUCLEO-N657X0-Q、完整 CubeMX 工程和串口 |
| 12 in-vivo shadow | NOT EXECUTED | 需要真实动物数据；禁止伪造结果 |

这里的 F37 9 次、F26 6 次是所有压力 candidate 的 T0 回放计数，并非 NVC
特异性计数。冻结离线报告中的 NVC 检出仍为 F37 6/6、F26 3/3；模型不能
特异区分 NVC 与 PREVOID。

## 6. 数值误差

- 模型：F37 `5.96e-08`，F26 `1.19e-07`。
- 完整回放 score：`2.98e-07`。
- 时域特征主要为 float32 rounding；最大 `p_auc_growth` 绝对误差
  `1.22e-04`，相对误差约 `5.90e-08`。
- 三个频域特征最大绝对误差分别约 `4.74e-07`、`2.38e-07`、`2.96e-08`。
- Candidate onset 完全一致；F26 B14/B15 各一个 recovery end 晚 0.01 s。
- T0 trigger count/event/timing 在 registered 0.25 s grid 上一致。

## 7. Host 测试

完整命令：

```powershell
powershell -ExecutionPolicy Bypass -File Test/v5_host/run_host_validation.ps1
```

已有 Golden 文件时：

```powershell
powershell -ExecutionPolicy Bypass -File Test/v5_host/run_host_validation.ps1 -UseExistingGolden
```

测试生成物保存在 `Test/v5_results/generated/`，并由 `.gitignore` 排除。

V5 Python 专项测试为 5/5 PASS。联合运行旧 detector 测试时另有 19 项 PASS、
3 项因先前已删除的 `data/STxF30_cycles` / `STxF30_nvc_results` fixture 无法运行；
这三项不是本次 C 移植回归失败，也未为通过测试而恢复或伪造旧数据。

## 8. STM32/HIL 准备状态

已提供 board-neutral `AppV5_On100Hz()`、二进制压力/telemetry 协议、CRC、
UART/USB transport adapter 入口和 PC replay 工具。HIL telemetry 包含 candidate、
15维特征、score、threshold、T0、latch、shadow/stimulation 状态及处理时间。

当前仓库只有 `VNS_N6.ioc`，没有完整的 `main/startup/HAL/linker/system` 工程骨架，
本环境也没有 STM32CubeMX、STM32CubeIDE 或 arm-none-eabi-gcc。因此没有执行
板级编译与 HIL，不能声称 100 Hz deadline 已通过。

真实板卡可用后的命令：

```powershell
python Test/v5_hil/replay_hil.py --port COMx --subject STxF37 --realtime
python Test/v5_hil/replay_hil.py --port COMx --subject STxF26 --realtime
```

验收要求：无丢 sample、score error `<1e-5`、T0 event/count 一致、timing
`<=0.25 s`（目标 exact）、最大处理时间 `<10 ms`。

## 9. 未完成与禁止启用刺激原因

第一处未完成的是 Gate 11：STM32 HIL。还缺：

1. 从 `VNS_N6.ioc` 恢复/生成完整 NUCLEO-N657X0-Q 工程骨架；
2. 接入板上 100 Hz pressure source 与 UART/USB transport；
3. 如 portable DFT 不满足 deadline，用 CMSIS-DSP 替换并重新做 feature parity；
4. F37/F26 HIL 的平均/最大运行时间和无丢样验证；
5. 真实动物 shadow logging 与安全审计。

因此当前刺激状态必须是 `DISABLED / SHADOW`，不是
`READY_FOR_CONTROLLED_VNS_ENABLE`。
