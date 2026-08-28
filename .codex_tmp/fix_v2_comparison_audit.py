import csv
from pathlib import Path


path = Path(r"D:\cubeIDE\project\VNS\data\dsd_validation\dsd_v1_vs_v2_comparison.csv")
with path.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle)); fields = list(rows[0])
for row in rows:
    if row["subject"] == "STxF29":
        row["change_reason"] = "BASELINE_SELECTION_CORRECTED;PHASIC_SEGMENTATION_NOT_CONFIRMED_RAW"
    elif row["subject"] == "STxF30":
        row["baseline_cycles_changed"] = "NOT_APPLICABLE"
        row["change_reason"] = "NO_CHANGE"
with path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
