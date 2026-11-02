"""Regenerate the published EP comparison figures from the adjacent CSV files."""

import csv
from pathlib import Path
from statistics import mean

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent


def save(figure, name):
    output = ROOT / "assets"
    output.mkdir(exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=160, metadata={"Software": "VeOmni validation"})
    figure.savefig(output / f"{name}.svg", metadata={"Creator": "VeOmni validation", "Date": None})
    svg = output / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    with (ROOT / "data/ep_timing.csv").open() as stream:
        timing = list(csv.DictReader(stream))
    with (ROOT / "data/ep_rank_load.csv").open() as stream:
        loads = list(csv.DictReader(stream))

    with rc_context({"svg.hashsalt": "veomni-ep-validation"}):
        figure = Figure(figsize=(10.5, 4.5), layout="constrained")
        FigureCanvasAgg(figure)
        left, right = figure.subplots(1, 2)
        for offset, mode, color in [(-0.24, "off", "#64748b"), (0, "legacy", "#059669"), (0.24, "memory", "#2563eb")]:
            values = [
                float(
                    next(
                        row["seconds_per_step"]
                        for row in timing
                        if row["cohort"] == "gp06" and row["mode"] == mode and int(row["group"]) == group
                    )
                )
                for group in range(5)
            ]
            left.bar([group + offset for group in range(5)], values, width=0.23, label=mode, color=color)
        left.set(title="GPU: all five paired groups retained", xlabel="Paired group", ylabel="Seconds per step")
        left.set_xticks(range(5), range(1, 6))
        left.set_ylim(0, 8)
        left.legend()
        npu = {row["mode"]: float(row["seconds_per_step"]) for row in timing if row["cohort"] == "npu-current"}
        bars = right.bar(["OFF", "Memory"], [npu["off"], npu["memory"]], color=["#64748b", "#2563eb"])
        right.bar_label(bars, fmt="%.3f s", padding=4)
        right.set(title="NPU: precision aligned; throughput regresses", ylabel="Seconds per step", ylim=(0, 8.5))
        figure.supxlabel(
            "GPU: 100 steps, 20 warm-up; NPU: 20 steps, 5 warm-up. Different configurations; compare within each panel.",
            fontsize=8,
        )
        save(figure, "ep_throughput")

        figure = Figure(figsize=(9, 4.5), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for offset, key, color in [(-0.18, "before", "#64748b"), (0.18, "after", "#2563eb")]:
            values = [mean(int(row[key]) for row in loads if int(row["rank"]) == rank) for rank in range(8)]
            axis.bar([rank + offset for rank in range(8)], values, width=0.35, label=key, color=color)
        axis.set(
            title="NPU: actual native-routing load redistribution",
            xlabel="EP rank",
            ylabel="Mean expert-token items per plan",
        )
        axis.set_xticks(range(8))
        axis.legend()
        figure.supxlabel(
            "480 plans across 24 layers and 20 steps; including warm-up for load counts only. Forward and input-gradient loads.",
            fontsize=8,
        )
        save(figure, "ep_rank_balance")


if __name__ == "__main__":
    main()
