"""Compare scalar metrics from all experiment runs into one Excel workbook.

Usage: python compare_runs.py --runs runs --output experiment_comparison.xlsx
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def flatten(obj, prefix=""):
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten(value, name))
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        result[prefix] = obj
    return result


def read_run(run_dir):
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = flatten(payload.get("metrics", {}))
            config = flatten(payload.get("config", {}))
            return {"log_filename": f"{run_dir.name}/log.txt", **metrics}, {"log_filename": f"{run_dir.name}/log.txt", **config}
        except (OSError, json.JSONDecodeError):
            pass
    # Fallback for runs where only epoch_metrics.jsonl exists: use final epoch.
    epoch_file = run_dir / "epoch_metrics.jsonl"
    if epoch_file.exists():
        records = []
        for line in epoch_file.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if records:
            last = records[-1]
            return {"log_filename": f"{run_dir.name}/log.txt", **flatten(last)}, {"log_filename": f"{run_dir.name}/log.txt"}
    return None, None


def collect_runs(root):
    metrics_rows, config_rows = [], []
    for summary in sorted(Path(root).rglob("summary.json")):
        row, config = read_run(summary.parent)
        if row is not None:
            metrics_rows.append(row)
            config_rows.append(config)
    # Include folders with epoch logs but no summary.
    for epoch_file in sorted(Path(root).rglob("epoch_metrics.jsonl")):
        if (epoch_file.parent / "summary.json").exists():
            continue
        row, config = read_run(epoch_file.parent)
        if row is not None:
            metrics_rows.append(row)
            config_rows.append(config)
    return metrics_rows, config_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="experiment_comparison.xlsx")
    args = parser.parse_args()
    metrics_rows, config_rows = collect_runs(args.runs)
    if not metrics_rows:
        raise SystemExit(f"No summary.json or epoch_metrics.jsonl found under {args.runs!r}")
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame(config_rows).to_excel(writer, sheet_name="Configs", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cells in sheet.columns:
                width = max((len(str(cell.value)) for cell in cells[:200] if cell.value is not None), default=10)
                sheet.column_dimensions[cells[0].column_letter].width = min(max(width + 2, 12), 45)
    print(f"Saved {args.output} with {len(metrics_rows)} runs.")

if __name__ == "__main__":
    main()
