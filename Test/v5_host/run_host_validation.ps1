param(
    [string]$GccPath = "D:\vscode\mingw64\bin\gcc.exe",
    [switch]$UseExistingGolden
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Build = Join-Path $Root "Test\v5_results\generated\build"
$Results = Join-Path $Root "Test\v5_results\generated"
$Include = Join-Path $Root "Modules\Inc"
New-Item -ItemType Directory -Force -Path $Build, $Results | Out-Null

if (-not (Test-Path -LiteralPath $GccPath)) {
    throw "GCC not found: $GccPath"
}

Push-Location $Root
try {
    if (-not $UseExistingGolden) {
        python Test/v5_host/generate_model_parity_vectors.py STxF26
        python Test/v5_host/generate_candidate_golden.py
        python Test/v5_host/generate_feature_golden.py
        python Test/v5_host/generate_full_replay_golden.py
    }

    function Build-C([string]$Name, [string[]]$Sources) {
        & $GccPath -std=c11 -O2 -Wall -Wextra -Werror -I $Include @Sources -lm -o (Join-Path $Build "$Name.exe")
        if ($LASTEXITCODE -ne 0) { throw "compile failed: $Name" }
    }

    Build-C "v5_f37_model" @("Test/v5_host/test_v5_f37_model.c", "Modules/Src/v5_model.c", "Modules/Src/v5_runtime.c", "Modules/Src/v5_stim_fsm.c")
    Build-C "v5_f26_model" @("Test/v5_host/test_v5_f26_model.c", "Modules/Src/v5_model.c", "Modules/Src/v5_runtime.c", "Modules/Src/v5_stim_fsm.c")
    Build-C "v5_runtime" @("Test/v5_host/test_v5_runtime_parity.c", "Modules/Src/v5_model.c", "Modules/Src/v5_runtime.c")
    Build-C "v5_candidate" @("Test/v5_host/test_v5_candidate_parity.c", "Modules/Src/v5_candidate.c")
    Build-C "v5_features" @("Test/v5_host/test_v5_feature_parity.c", "Modules/Src/v5_features.c")
    Build-C "v5_full" @("Test/v5_host/test_v5_full_replay.c", "Modules/Src/v5_candidate.c", "Modules/Src/v5_features.c", "Modules/Src/v5_model.c", "Modules/Src/v5_runtime.c", "Modules/Src/v5_subject_config.c")
    Build-C "v5_safety" @("Test/v5_host/test_v5_safety.c", "Core/Src/v5_integration_example.c", "Modules/Src/v5_candidate.c", "Modules/Src/v5_features.c", "Modules/Src/v5_model.c", "Modules/Src/v5_runtime.c", "Modules/Src/v5_stim_fsm.c", "Modules/Src/v5_subject_config.c")

    & (Join-Path $Build "v5_f37_model.exe") | Tee-Object (Join-Path $Results "f37_model.txt")
    & (Join-Path $Build "v5_f26_model.exe") | Tee-Object (Join-Path $Results "f26_model.txt")
    & (Join-Path $Build "v5_runtime.exe") data/NVC_V5/v5_final_validation/m1_full_cycle_replay.csv | Tee-Object (Join-Path $Results "runtime.txt")
    & (Join-Path $Build "v5_candidate.exe") Test/v5_results/generated/candidate_golden.csv | Tee-Object (Join-Path $Results "candidate.txt")
    & (Join-Path $Build "v5_features.exe") Test/v5_results/generated/feature_golden.csv | Tee-Object (Join-Path $Results "features.txt")
    & (Join-Path $Build "v5_full.exe") Test/v5_results/generated/full_replay_golden.csv data/NVC_V5/mcu_config/STxF37_v5_subject_config.bin data/NVC_V5/mcu_config/STxF26_v5_subject_config.bin | Tee-Object (Join-Path $Results "full_replay.txt")
    & (Join-Path $Build "v5_safety.exe") data/NVC_V5/mcu_config/STxF37_v5_subject_config.bin | Tee-Object (Join-Path $Results "safety.txt")
    if ($LASTEXITCODE -ne 0) { throw "V5 host validation failed" }
    Write-Output "PASS_V5_HOST_VALIDATION"
}
finally {
    Pop-Location
}
