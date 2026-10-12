"""Compare metrics from all experiment runs into an Excel workbook.

Usage:
    python compare_runs.py --runs runs --output experiment_comparison.xlsx
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def flatten(obj, prefix=""):
    """Flatten nested dictionaries into scalar Excel columns."""
    result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten(value, name))

    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        if isinstance(obj, float) and not math.isfinite(obj):
            result[prefix] = None
        else:
            result[prefix] = obj

    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            name = f"{prefix}/{i}" if prefix else str(i)
            result.update(flatten(value, name))

    return result


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_run(run_dir):
    run_name = run_dir.name
    log_filename = f"{run_name}/log.txt"
    summary_path = run_dir / "summary.json"

    if summary_path.exists():
        payload = read_json(summary_path)

        if isinstance(payload, dict):
            # Support both nested and direct metric structures.
            metrics = payload.get("metrics", {})
            config = payload.get("config", {})

            # Include test metrics if stored separately in the summary.
            test_metrics = payload.get("test_metrics", {})
            if isinstance(test_metrics, dict):
                metrics = {
                    **(metrics if isinstance(metrics, dict) else {}),
                    **{
                        f"test/{key}": value
                        for key, value in test_metrics.items()
                    },
                }

            metric_row = {
                "run": run_name,
                "log_filename": log_filename,
                **flatten(metrics if isinstance(metrics, dict) else {}),
            }

            config_row = {
                "run": run_name,
                "log_filename": log_filename,
                **flatten(config if isinstance(config, dict) else {}),
            }

            return metric_row, config_row

    # Fallback: use the final record in epoch_metrics.jsonl.
    epoch_file = run_dir / "epoch_metrics.jsonl"

    if epoch_file.exists():
        records = []

        for line in epoch_file.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if records:
            return (
                {
                    "run": run_name,
                    "log_filename": log_filename,
                    **flatten(records[-1]),
                },
                {
                    "run": run_name,
                    "log_filename": log_filename,
                },
            )

    return None, None


def collect_runs(root):
    metrics_rows = []
    config_rows = []

    root = Path(root)

    run_dirs = sorted({
        path.parent
        for pattern in ("summary.json", "epoch_metrics.jsonl")
        for path in root.rglob(pattern)
    })

    for run_dir in run_dirs:
        metrics, config = read_run(run_dir)

        if metrics is not None:
            metrics_rows.append(metrics)
            config_rows.append(config)

    return metrics_rows, config_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument(
        "--output",
        default="experiment_comparison.xlsx",
    )
    args = parser.parse_args()

    metrics_rows, config_rows = collect_runs(args.runs)

    if not metrics_rows:
        raise SystemExit(
            f"No summary.json or epoch_metrics.jsonl found under {args.runs!r}"
        )

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        pd.DataFrame(metrics_rows).to_excel(
            writer, sheet_name="Metrics", index=False
        )
        pd.DataFrame(config_rows).to_excel(
            writer, sheet_name="Configs", index=False
        )

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions

            for cells in sheet.columns:
                width = max(
                    (
                        len(str(cell.value))
                        for cell in cells[:200]
                        if cell.value is not None
                    ),
                    default=10,
                )
                sheet.column_dimensions[
                    cells[0].column_letter
                ].width = min(max(width + 2, 12), 45)

    print(
        f"Saved {args.output} with {len(metrics_rows)} runs."
    )


if __name__ == "__main__":
    main()
