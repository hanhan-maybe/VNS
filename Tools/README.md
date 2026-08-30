# SPARC338刺激前数据与DSD排尿周期提取

## 1. 模块范围

本模块用于SPARC Dataset 338的前两阶段离线处理：

1. 从SCI大鼠`.smrx`文件中提取第一次VNS刺激前的数据，输出至`data/baseline`。
2. 从已确认DSD表型的动物中提取稳定排尿周期，输出至`data/DSD_cycles`。

当前不运行排尿前事件普查，不生成在线VNS触发事件，也不执行真实刺激。

---

## 2. 当前目录要求

主文件应直接放在项目的`Tools`目录。推荐结构：

```text
D:\cubeIDE\project\VNS\
├── data\
│   ├── baseline\
│   ├── DSD_cycles\
│   └── dsd_validation\
└── Tools\
    ├── README.md
    ├── rerun_two_stage_pipeline.py
    ├── sparc338_common.py
    ├── sparc338_config.py
    ├── sparc338_pre_stim_extract.py
    ├── sparc338_pre_stim_qc.py
    ├── sparc338_preprocessing.py
    ├── sparc338_smrx_reader.py
    ├── sparc338_stable_phase.py
    ├── sparc338_urine_output.py
    └── dsd_cycle_extraction\
        ├── __init__.py
        ├── config.py
        ├── cycle_qc.py
        ├── pipeline.py
        ├── plot_cycles.py
        ├── stable_cycle_extractor.py
        ├── urine_evidence_adapter.py
        └── test_cycle_extraction.py
```

### 第二阶段文件夹名称

当前代码中的总入口使用：

```python
python -m Tools.dsd_cycle_extraction.pipeline
```

因此第二阶段文件夹必须命名为：

```text
dsd_cycle_extraction
```

如果当前文件夹名称是：

```text
sparc338_dsd_cycle_extraction
```

且没有同步修改Python导入路径，执行总入口时会出现：

```text
No module named Tools.dsd_cycle_extraction
```

最简单的处理方式是将文件夹重命名为`dsd_cycle_extraction`。本说明后续均按该名称书写。

---

## 3. 文件职责

| 文件 | 主要职责 | 是否直接运行 |
|---|---|---|
| `sparc338_config.py` | 项目路径、SCI/DSD动物名单、尿液证据注册表及公共参数 | 否 |
| `sparc338_common.py` | CSV/JSON原子写入、SHA256、临时目录和安全提交 | 否 |
| `sparc338_smrx_reader.py` | 只读打开SMRX、扫描和匹配通道、读取事件与波形 | 否 |
| `sparc338_preprocessing.py` | CMG/EUS因果滤波、100 Hz降采样、非有限值处理 | 否 |
| `sparc338_urine_output.py` | Volume或Leaks尿液证据的离线解析和QC | 否 |
| `sparc338_pre_stim_qc.py` | 第一阶段QC、summary和quicklook | 否 |
| `sparc338_pre_stim_extract.py` | 第一阶段入口：提取第一次刺激前数据 | 是 |
| `sparc338_stable_phase.py` | CMG＋尿液证据确认排尿并生成稳定性候选 | 否 |
| `dsd_cycle_extraction/pipeline.py` | 第二阶段入口：建立DSD稳定周期数据集 | 是 |
| `rerun_two_stage_pipeline.py` | 按顺序连续运行第一、第二阶段 | 是 |

`test_*.py`属于回归测试，不是日常运行入口。建议保留并集中放到`Tools/tests`，但移动测试前需要同步修改其导入路径。

---

## 4. 数据流程

```text
SCI动物SMRX
    │
    ├── 识别CMG、native EUS、Stim、Volume/Leaks、Keyboard
    ├── 确定第一次刺激串起点 first_stim_s
    ├── 严格截取 0 ≤ t < first_stim_s
    ├── 保存native信号
    └── 因果处理并降采样至100 Hz
            │
            ▼
       data/baseline
            │
            ├── CMG收缩候选
            ├── 同步尿液证据确认排尿
            ├── 相邻排尿构建完整周期
            └── 稳定性、伪迹和数据缺口QC
                    │
                    ├── 稳定分析周期
                    └── 参考基线周期
                            │
                            ▼
                     data/DSD_cycles
```

---

## 5. 第一阶段：刺激前数据提取

### 5.1 输入数据

默认原始数据目录：

```text
D:\Sparc
```

SCI动物文件：

```text
STxF14.smrx
STxF21.smrx
STxF22.smrx
STxF23.smrx
STxF24.smrx
STxF26.smrx
STxF27.smrx
STxF29.smrx
STxF30.smrx
```

必须识别的通道：

- CMG/膀胱压力；
- native EUS；
- Stim事件。

可选通道包括Volume、Leaks和Keyboard。缺少必须通道或无法检测第一次刺激串时，该动物状态为`FAIL`。

### 5.2 处理原则

- 第一次刺激串起点定义为`first_stim_s`。
- 所有输出样本严格满足`0 <= t < first_stim_s`。
- native CMG和native EUS无损保留。
- CMG与EUS预处理只使用当前及历史样本，不读取未来样本。
- EUS经过带通、整流、包络提取和100 Hz降采样。
- 非有限值采用因果前向保持，同时保存有效性mask。
- Volume/Leaks只用于离线确认和QC，不作为在线刺激输入。

### 5.3 运行命令

在项目根目录执行：

```powershell
cd D:\cubeIDE\project\VNS
python Tools\sparc338_pre_stim_extract.py
```

显式指定路径：

```powershell
python Tools\sparc338_pre_stim_extract.py `
  --raw-dir D:\Sparc `
  --output-dir D:\cubeIDE\project\VNS\data\baseline
```

仅处理指定动物：

```powershell
python Tools\sparc338_pre_stim_extract.py --subjects STxF21 STxF26
```

最终冻结数据时可计算原始SMRX的SHA256：

```powershell
python Tools\sparc338_pre_stim_extract.py --hash-source
```

### 5.4 第一阶段输出

每只动物输出至：

```text
data\baseline\<subject>\
```

主要文件：

| 文件 | 内容 |
|---|---|
| `pre_stim_raw.npz` | native CMG/EUS、采样率、单位、事件和native有效性mask |
| `pre_stim_100Hz.npz` | 100 Hz CMG、EUS包络、时间轴及有效性mask |
| `pre_stim_summary.json` | 第一次刺激时间、QC、信号缺口和来源信息 |
| `channel_inventory.csv` | SMRX通道清单及匹配结果 |
| `stim_trains.csv` | 刺激串时间、脉冲数和频率 |
| `pre_stim_events.csv` | Keyboard和Leaks等离散事件 |
| `urine_output_info.json` | 尿液证据来源及审核状态 |
| `pre_stim_urine_output.npz` | 连续Volume原始信号，若存在 |
| quicklook/QC图片 | 人工核对CMG、EUS、尿液信号及边界 |

总汇文件：

```text
data\baseline\pre_stim_inventory.csv
data\baseline\volume_channel_qc.csv
```

---

## 6. 第二阶段：DSD排尿周期提取

### 6.1 动物范围

当前只处理：

```text
STxF21
STxF26
STxF27
STxF29
```

本阶段读取既定DSD名单，不重新判定DSD表型。DSD表型审核仍需结合native-rate EUS形态。

### 6.2 确认排尿

确认排尿必须同时存在：

1. CMG收缩候选；
2. 与收缩同步的已审核尿液证据。

只有压力峰、没有尿液证据时不能定义为确认排尿。

当前尿液证据：

- STxF21：channel 5 `Leaks`按钮上升沿；
- STxF26、STxF27、STxF29：连续Weight/Volume信号；
- Keyboard保留为metadata，不在本次重构中改变既有时间定义。

### 6.3 周期边界

周期定义为：

```text
前一次确认排尿的稳定结束点 → 当前确认排尿的稳定结束点
```

周期长度由真实相邻排尿决定，不使用固定窗口。

### 6.4 两类周期

#### 稳定分析周期

从最早得到连续3个稳定候选支持的稳定起点开始，纳入之后全部满足QC的`PASS_STABLE`周期。

主要清单：

```text
data\DSD_cycles\cycle_manifest.csv
```

#### 参考基线周期

从距离第一次刺激最近的稳定连续段中选择末尾3–5个周期，用于后续个体/session基线初始化参考，不改变分析周期总数。

主要输出：

```text
data\DSD_cycles\reference_baseline_manifest.csv
data\DSD_cycles\reference_baseline_stats.csv
```

`reference_baseline_stats.csv`中的中位数和MAD只能作为离线参考。新动物或新实验session必须重新建立当次因果基线，不能直接把338数据的数值固定为在线阈值。

### 6.5 运行命令

```powershell
cd D:\cubeIDE\project\VNS
python -m Tools.dsd_cycle_extraction.pipeline
```

显式指定路径：

```powershell
python -m Tools.dsd_cycle_extraction.pipeline `
  --baseline-root D:\cubeIDE\project\VNS\data\baseline `
  --output-root D:\cubeIDE\project\VNS\data\DSD_cycles
```

### 6.6 第二阶段输出

| 文件 | 内容 |
|---|---|
| `cycle_manifest.csv` | 全部纳入的稳定DSD分析周期 |
| `nvc_cycle_manifest.csv` | 通过完整性、排尿确认、数据缺失、伪迹、刺激边界与最短时长硬QC的NVC可判断周期；统计稳定性异常保留为复核标记 |
| `reference_baseline_manifest.csv` | 每只动物末尾3–5个参考周期 |
| `reference_baseline_stats.csv` | 参考周期的个体中位数与MAD |
| `all_candidate_cycles.csv` | 所有周期及纳入/排除原因 |
| `all_confirmed_voids.csv` | CMG＋尿液证据确认的排尿 |
| `subject_summary.csv` | 每只动物的排尿数、分析周期数和参考周期数 |
| `pipeline_contract.json` | 动物范围、信号角色及周期定义 |
| `generated_output_validation.json` | 输出文件和边界检查结果 |
| `source_integrity.json` | 输入文件运行前后SHA256比较 |
| `extraction_report.md` | 第二阶段汇总报告 |

每个纳入周期位于：

```text
data\DSD_cycles\<subject>\Bxx\
```

包含：

- `cycle_100Hz.npz`；
- `cycle_native_eus.npz`；
- `quicklook.png`。

### 6.7 排除状态

常见状态包括：

| 状态 | 含义 |
|---|---|
| `EXCLUDE_INCOMPLETE` | 无前一确认排尿，无法构成完整周期 |
| `EXCLUDE_ACCLIMATION` | 位于稳定起点之前 |
| `EXCLUDE_DATA_GAP` | CMG无效样本比例超过周期的0.5% |
| `EXCLUDE_PRESSURE_ARTIFACT` | 严重压力范围或突变异常 |
| `EXCLUDE_TRANSITIONAL` | 稳定性QC不通过 |
| `EXCLUDE_PRE_STIM_BOUNDARY` | 周期跨越第一次刺激边界 |
| `PASS_STABLE` | 纳入稳定分析周期 |

EUS缺失比例会被记录，但不参与本阶段稳定周期选择。

稳定基线与NVC准入采用两级口径：`cycle_manifest.csv`仍要求持续稳定起点，
用于严格稳定基线；`nvc_cycle_manifest.csv`按单周期硬QC独立准入。孤立稳定候选或仅有
个体内统计离群的完整确认排尿周期不会因此整段丢弃，而会标记
`NVC_ELIGIBLE_STATISTICAL_REVIEW`。完整周期少于5个时，个体内robust-z仅记录，
不作为稳定性一票否决条件。

---

## 7. 两阶段连续运行

完整运行：

```powershell
cd D:\cubeIDE\project\VNS
python Tools\rerun_two_stage_pipeline.py
```

如果`baseline`已完成且人工确认，只重新提取DSD周期：

```powershell
python Tools\rerun_two_stage_pipeline.py --skip-baseline
```

推荐第一次部署重构代码时先运行`--skip-baseline`，确认第二阶段仍得到预期的40个DSD分析周期，再决定是否从原始SMRX完整重建。

---

## 8. 数据安全和重跑规则

### 第一阶段

每只动物先在临时目录生成，成功后再提交到`baseline/<subject>`。单只动物失败时，不提交其不完整目录。

### 第二阶段

整个`DSD_cycles`先在同级临时目录生成，通过以下检查后才提交：

- manifest中的所有周期文件存在；
- 100 Hz和native EUS均严格早于第一次刺激；
- 参考基线标志与manifest一致；
- baseline和validation输入SHA256未改变。

事务式替换可以避免旧B编号周期残留，但会用新结果替换同名`DSD_cycles`。正式重跑前仍建议备份当前已人工确认的：

```text
baseline
DSD_cycles
```

原始`.smrx`必须只读保存。

---

## 9. 软件依赖

```text
numpy
scipy
matplotlib
sonpy
```

SonPy只在读取`.smrx`时需要。建议继续使用已经能够读取Dataset338的Python/SonPy环境。

基础检查：

```powershell
python --version
python -c "import numpy, scipy, matplotlib; print('basic dependencies: PASS')"
python -c "import sonpy; print('SonPy: PASS')"
```

---

## 10. 运行后检查清单

### 第一阶段

- 9只SCI动物是否均生成`pre_stim_summary.json`；
- `pre_stim_inventory.csv`是否存在`FAIL`；
- `first_stim_s`是否对应真实第一次刺激串；
- quicklook是否严格截止在刺激前；
- STxF21 Leaks事件是否与原审核一致；
- Volume通道审核结论是否保持一致。

### 第二阶段

- 是否只包含STxF21、STxF26、STxF27和STxF29；
- `cycle_manifest.csv`是否仍为预期40个分析周期；
- 每只动物B编号是否连续；
- 每只动物参考基线是否为3–5个周期或明确显示不足；
- `generated_output_validation.json`是否通过；
- `source_integrity.json`中的`sha256_all_identical`是否为`true`；
- 周期quicklook及native EUS切片是否正确。

如果分析周期不再是40，应暂停后续分析，依次比较：

1. 新旧`cycle_manifest.csv`；
2. `all_candidate_cycles.csv`中的排除原因；
3. CMG有效性mask和`EXCLUDE_DATA_GAP`；
4. 尿液证据确认结果；
5. 第一次刺激时间和周期结束边界。

---

## 11. 与后续在线VNS的关系

本模块已经保留：

- 因果CMG/EUS预处理语义；
- 每只动物的参考基线周期和统计量；
- CMG/EUS信号有效性mask；
- native EUS用于DSD和排尿形态审核；
- 尿液证据只用于离线确认的边界。

后续在线系统仍需独立实现：

- 每只动物、每次实验重新初始化的因果滚动基线；
- Class2/排尿优先保护；
- 排尿、伪迹和恢复期刺激抑制；
- 每个合格事件最多触发一次；
- 延迟、刺激负担和失败原因记录。

当前模块输出不能直接驱动真实VNS刺激。

---

## 12. 推荐执行顺序

```text
1. 确认主文件位于VNS\Tools
2. 将第二阶段文件夹命名为dsd_cycle_extraction
3. 备份当前baseline和DSD_cycles
4. 先运行rerun_two_stage_pipeline.py --skip-baseline
5. 核对40个分析周期、参考基线和完整性报告
6. 确认第二阶段无误后，再决定是否完整读取SMRX
7. 冻结最终输出、配置、报告和SHA256
8. 后续再开展排尿前事件标注/普查和在线算法开发
```
