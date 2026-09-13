# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import sqlite3
from collections import defaultdict
from statistics import mean


STAGES = ("backward_combine", "replay_dispatch", "backward_dispatch", "replay_combine")


def _steady(values: list, warmup_steps: int) -> list:
    steady = values[warmup_steps:]
    if not steady:
        raise RuntimeError(f"No steady samples remain after excluding {warmup_steps} warmup steps.")
    return steady


def _step_mean(path: str, *, warmup_steps: int) -> float:
    with sqlite3.connect(path) as db:
        steps = db.execute("select startNs, endNs from STEP_TIME order by id").fetchall()
    return mean((end - start) / 1e6 for start, end in _steady(steps, warmup_steps))


def _step_call_intervals(path: str, *, num_layers: int, warmup_steps: int) -> list[list[tuple[int, int]]]:
    expected_calls = 6 * num_layers
    with sqlite3.connect(path) as db:
        steps = db.execute("select startNs, endNs from STEP_TIME order by id").fetchall()
        calls = []
        for start_ns, end_ns in _steady(steps, warmup_steps):
            rows = db.execute(
                """
                select communication.startNs, communication.endNs
                from COMMUNICATION_OP communication
                join STRING_IDS name on name.id = communication.opName
                where communication.startNs >= ? and communication.endNs <= ?
                  and lower(name.value) like 'hcom_alltoall%'
                order by communication.startNs
                """,
                (start_ns, end_ns),
            ).fetchall()
            if len(rows) != expected_calls:
                raise RuntimeError(
                    f"Expected {expected_calls} AllToAll calls per step for {num_layers} layers in {path}, "
                    f"found {len(rows)}."
                )
            calls.append(rows)
    return calls


def _step_calls(path: str, *, num_layers: int, warmup_steps: int) -> list[list[float]]:
    intervals = _step_call_intervals(path, num_layers=num_layers, warmup_steps=warmup_steps)
    return [[(end - start) / 1e6 for start, end in calls] for calls in intervals]


def _semantic_steps(
    path: str,
    *,
    ilp: bool,
    num_layers: int,
    current_layer: int,
    window_size: int,
    warmup_steps: int,
) -> list[dict[str, float]]:
    semantic_steps = []
    boundary_count = num_layers - 1
    resolved_current_layer = num_layers - 1 if current_layer < 0 else current_layer
    active_boundaries = resolved_current_layer if window_size == 0 else window_size
    bottom_layer = resolved_current_layer - active_boundaries
    for calls in _step_calls(path, num_layers=num_layers, warmup_steps=warmup_steps):
        values = {
            "original_forward": sum(calls[: 2 * num_layers]),
            "top_recompute": sum(calls[2 * num_layers : 2 * num_layers + 2]),
            "bottom_backward": sum(calls[-2:]),
        }
        boundary_offset = 2 * num_layers + 2
        for boundary in range(boundary_count):
            layer = num_layers - 1 - boundary
            group = calls[boundary_offset + boundary * 4 : boundary_offset + (boundary + 1) * 4]
            if ilp and bottom_layer < layer <= resolved_current_layer:
                mapped = dict(zip(STAGES, group, strict=True))
            else:
                mapped = {
                    "backward_combine": group[0],
                    "replay_dispatch": group[2],
                    "backward_dispatch": group[1],
                    "replay_combine": group[3],
                }
            for stage, duration in mapped.items():
                values[f"B{layer}/{stage}"] = duration
        semantic_steps.append(values)
    return semantic_steps


def _means(steps: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(step[key] for step in steps) for key in steps[0]}


def _boundary_intervals(
    path: str,
    *,
    ilp: bool,
    num_layers: int,
    current_layer: int,
    window_size: int,
    warmup_steps: int,
) -> list[dict[str, tuple[int, int]]]:
    semantic_steps = []
    boundary_count = num_layers - 1
    resolved_current_layer = num_layers - 1 if current_layer < 0 else current_layer
    active_boundaries = resolved_current_layer if window_size == 0 else window_size
    bottom_layer = resolved_current_layer - active_boundaries
    intervals = _step_call_intervals(path, num_layers=num_layers, warmup_steps=warmup_steps)
    for calls in intervals:
        values = {}
        boundary_offset = 2 * num_layers + 2
        for boundary in range(boundary_count):
            layer = num_layers - 1 - boundary
            group = calls[boundary_offset + boundary * 4 : boundary_offset + (boundary + 1) * 4]
            if ilp and bottom_layer < layer <= resolved_current_layer:
                mapped = dict(zip(STAGES, group, strict=True))
            else:
                mapped = {
                    "backward_combine": group[0],
                    "replay_dispatch": group[2],
                    "backward_dispatch": group[1],
                    "replay_combine": group[3],
                }
            for stage, interval in mapped.items():
                values[f"B{layer}/{stage}"] = interval
        semantic_steps.append(values)
    return semantic_steps


def _rank_metrics(
    paths: list[str],
    *,
    ilp: bool,
    num_layers: int,
    current_layer: int,
    window_size: int,
    warmup_steps: int,
) -> dict[str, tuple[float, float, float]]:
    ranks = [
        _boundary_intervals(
            path,
            ilp=ilp,
            num_layers=num_layers,
            current_layer=current_layer,
            window_size=window_size,
            warmup_steps=warmup_steps,
        )
        for path in paths
    ]
    step_counts = {len(steps) for steps in ranks}
    if len(step_counts) != 1:
        raise RuntimeError(f"Rank profiler DBs have different step counts: {sorted(step_counts)}")

    samples = defaultdict(list)
    for step_index in range(len(ranks[0])):
        for key in ranks[0][step_index]:
            intervals = [rank[step_index][key] for rank in ranks]
            starts = [start for start, _ in intervals]
            ends = [end for _, end in intervals]
            mean_duration = mean((end - start) / 1e6 for start, end in intervals)
            arrival_spread = (max(starts) - min(starts)) / 1e6
            post_arrival_tail = (max(ends) - max(starts)) / 1e6
            samples[key].append((mean_duration, arrival_spread, post_arrival_tail))
    return {key: tuple(mean(values[index] for values in rows) for index in range(3)) for key, rows in samples.items()}


def _boundary_envelopes(
    paths: list[str],
    *,
    ilp: bool,
    num_layers: int,
    current_layer: int,
    window_size: int,
    warmup_steps: int,
) -> dict[int, float]:
    ranks = [
        _boundary_intervals(
            path,
            ilp=ilp,
            num_layers=num_layers,
            current_layer=current_layer,
            window_size=window_size,
            warmup_steps=warmup_steps,
        )
        for path in paths
    ]
    samples = defaultdict(list)
    for step_index in range(len(ranks[0])):
        for layer in range(num_layers - 1, 0, -1):
            intervals = [rank[step_index][f"B{layer}/{stage}"] for rank in ranks for stage in STAGES]
            samples[layer].append((max(end for _, end in intervals) - min(start for start, _ in intervals)) / 1e6)
    return {layer: mean(values) for layer, values in samples.items()}


def _print_rank_metrics(
    baseline_paths: list[str],
    ilp_paths: list[str],
    *,
    num_layers: int,
    current_layer: int,
    window_size: int,
    warmup_steps: int,
) -> None:
    if len(baseline_paths) != len(ilp_paths):
        raise RuntimeError("Baseline and ILP must contain the same number of rank DBs.")
    if len(baseline_paths) < 2:
        raise RuntimeError("Rank-skew analysis requires at least two DBs per mode.")

    metric_kwargs = {
        "num_layers": num_layers,
        "current_layer": current_layer,
        "window_size": window_size,
        "warmup_steps": warmup_steps,
    }
    baseline = _rank_metrics(baseline_paths, ilp=False, **metric_kwargs)
    ilp = _rank_metrics(ilp_paths, ilp=True, **metric_kwargs)
    print("\nall-rank stage totals")
    print("stage              duration_delta  arrival_spread_delta  post_arrival_tail_delta")
    for stage in STAGES:
        deltas = []
        for metric in range(3):
            deltas.append(
                sum(
                    ilp[f"B{layer}/{stage}"][metric] - baseline[f"B{layer}/{stage}"][metric]
                    for layer in range(num_layers - 1, 0, -1)
                )
            )
        print(f"{stage:<18} {deltas[0]:+14.3f} {deltas[1]:+21.3f} {deltas[2]:+24.3f}")

    rows = []
    for layer in range(num_layers - 1, 0, -1):
        for stage in STAGES:
            key = f"B{layer}/{stage}"
            duration_delta = ilp[key][0] - baseline[key][0]
            spread_delta = ilp[key][1] - baseline[key][1]
            tail_delta = ilp[key][2] - baseline[key][2]
            rows.append((duration_delta, layer, stage, spread_delta, tail_delta))
    print("\nlargest all-rank duration taxes")
    for duration_delta, layer, stage, spread_delta, tail_delta in sorted(rows, reverse=True)[:8]:
        print(
            f"B{layer}/F'{layer - 1} {stage:<18} duration {duration_delta:+7.3f} ms, "
            f"arrival {spread_delta:+7.3f} ms, tail {tail_delta:+7.3f} ms"
        )

    baseline_envelopes = _boundary_envelopes(baseline_paths, ilp=False, **metric_kwargs)
    ilp_envelopes = _boundary_envelopes(ilp_paths, ilp=True, **metric_kwargs)
    print("\nall-rank boundary envelopes")
    print("boundary  baseline_ms  ilp_ms  delta_ms")
    for layer in range(num_layers - 1, 0, -1):
        delta = ilp_envelopes[layer] - baseline_envelopes[layer]
        print(f"B{layer}/F'{layer - 1} {baseline_envelopes[layer]:11.3f} {ilp_envelopes[layer]:7.3f} {delta:+9.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute VeOmni ILP AllToAll overlap tax by layer and stage.")
    parser.add_argument("baseline_db")
    parser.add_argument("ilp_db")
    parser.add_argument("--baseline-peer-db", action="append", default=[])
    parser.add_argument("--ilp-peer-db", action="append", default=[])
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--current-layer", type=int, default=-1, help="-1 selects the final decoder layer.")
    parser.add_argument("--window-size", type=int, default=0, help="0 selects every inter-layer boundary.")
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()

    if args.num_layers < 2:
        parser.error("--num-layers must be at least 2")
    resolved_current_layer = args.num_layers - 1 if args.current_layer < 0 else args.current_layer
    if not 1 <= resolved_current_layer < args.num_layers:
        parser.error("--current-layer must be -1 or between 1 and num-layers - 1")
    if not 0 <= args.window_size <= resolved_current_layer:
        parser.error("--window-size must be between 0 and current-layer")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    semantic_kwargs = {
        "num_layers": args.num_layers,
        "current_layer": args.current_layer,
        "window_size": args.window_size,
        "warmup_steps": args.warmup_steps,
    }

    baseline = _means(_semantic_steps(args.baseline_db, ilp=False, **semantic_kwargs))
    ilp = _means(_semantic_steps(args.ilp_db, ilp=True, **semantic_kwargs))
    totals = defaultdict(float)

    baseline_step = _step_mean(args.baseline_db, warmup_steps=args.warmup_steps)
    ilp_step = _step_mean(args.ilp_db, warmup_steps=args.warmup_steps)
    print(
        f"step baseline {baseline_step:.3f} ms, ILP {ilp_step:.3f} ms, "
        f"delta {ilp_step - baseline_step:+.3f} ms ({(baseline_step - ilp_step) / baseline_step * 100:+.2f}%)\n"
    )

    print("boundary stage              baseline_ms  ilp_ms  delta_ms")
    rows = []
    for layer in range(args.num_layers - 1, 0, -1):
        for stage in STAGES:
            key = f"B{layer}/{stage}"
            delta = ilp[key] - baseline[key]
            totals[stage] += delta
            rows.append((delta, layer, stage, baseline[key], ilp[key]))
            print(f"B{layer}/F'{layer - 1:<1} {stage:<18} {baseline[key]:11.3f} {ilp[key]:7.3f} {delta:+9.3f}")

    print("\nstage totals")
    for stage in STAGES:
        print(f"{stage:<18} {totals[stage]:+9.3f} ms")
    edge_delta = sum(totals.values())
    non_edge_delta = sum(ilp[key] - baseline[key] for key in ("original_forward", "top_recompute", "bottom_backward"))
    print(f"cross-layer total  {edge_delta:+9.3f} ms")
    print(f"other total        {non_edge_delta:+9.3f} ms")
    print(f"alltoall total     {edge_delta + non_edge_delta:+9.3f} ms")

    print("\nlargest taxes")
    for delta, layer, stage, baseline_ms, ilp_ms in sorted(rows, reverse=True)[:8]:
        print(f"B{layer}/F'{layer - 1} {stage:<18} {delta:+8.3f} ms ({baseline_ms:.3f}->{ilp_ms:.3f})")

    if args.baseline_peer_db or args.ilp_peer_db:
        _print_rank_metrics(
            [args.baseline_db, *args.baseline_peer_db],
            [args.ilp_db, *args.ilp_peer_db],
            num_layers=args.num_layers,
            current_layer=args.current_layer,
            window_size=args.window_size,
            warmup_steps=args.warmup_steps,
        )


if __name__ == "__main__":
    main()
