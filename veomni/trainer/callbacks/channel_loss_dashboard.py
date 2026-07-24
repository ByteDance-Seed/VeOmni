# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact W&B dashboard for the channel-loss scalar contract.

The dashboard combines the framework-owned scalar metrics into one stable,
user-facing HTML panel. It deliberately has no dataloader, storage, or sample
content dependency.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ...utils.logging import get_logger
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer


logger = get_logger(__name__)

_DASHBOARD_KEY = "channel_overview"
_DEFAULT_MAX_RENDER_POINTS = 2000
_MAX_RETAINED_POINTS = 10000
_FOUNDATION_LOSS_KEY = "training/foundation_loss"


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning_once(f"Ignoring invalid {name}={raw_value!r}; expected a positive integer.")
        return default
    if value < 1:
        logger.warning_once(f"Ignoring invalid {name}={raw_value!r}; expected a positive integer.")
        return default
    return value


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stable_source_fragment(value: Any) -> str:
    """Mirror the stable part of the public metric name without parsing labels."""

    if isinstance(value, int):
        return f"source-i-{value}"
    encoded = str(value).encode("utf-8").hex()
    return f"source-s-{encoded or 'empty'}"


def _source_descriptors(trainer: BaseTrainer, metric_fragments: list[str]) -> dict[str, tuple[str, str]]:
    """Resolve stable keys and labels while containing the compatibility shim.

    The fallback uses the complete metric fragment for both fields.  This
    function never splits on ``__`` because a valid source name may contain that
    sequence.  When the core registry is available, the chart key is based only
    on the stable source ID, so a display-name change cannot split one source
    into multiple series.  If a future core version changes its registry
    internals, collection keeps working with less-pretty labels.
    """

    descriptors = {fragment: (fragment, fragment) for fragment in metric_fragments}
    callback = getattr(trainer, "channel_loss_callback", None)
    registry = getattr(callback, "_source_registry", None)
    if not isinstance(registry, Mapping):
        return descriptors

    for source_id, source_name in registry.items():
        stable_key = _stable_source_fragment(source_id)
        stable_prefix = f"{stable_key}__"
        matches = [fragment for fragment in metric_fragments if fragment.startswith(stable_prefix)]
        if len(matches) != 1:
            continue
        label = str(source_name) if source_name not in (None, "") else f"source {source_id}"
        descriptors[matches[0]] = (stable_key, label)
    return descriptors


def _metric_values(metrics: Mapping[str, Any], prefix: str) -> dict[str, float]:
    prefix_with_slash = f"{prefix}/"
    result: dict[str, float] = {}
    for name, raw_value in metrics.items():
        if not name.startswith(prefix_with_slash):
            continue
        value = _finite_float(raw_value)
        if value is not None:
            result[name[len(prefix_with_slash) :]] = value
    return result


def _sanitized_metric_fragment(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def _data_metric_suffixes(
    available_suffixes: set[str],
    fragments: list[str],
    descriptors: Mapping[str, tuple[str, str]],
) -> dict[str, str]:
    """Resolve every source suffix jointly without coupling to core internals.

    Data metrics use a short source label until two labels collide.  After a
    collision they use the same stable-qualified suffix as channel loss.  The
    adapter accepts both forms and never assigns one suffix to two sources.
    Friendly suffixes are resolved for all sources before stable fallbacks so a
    valid source label such as ``source-i-0__repoqa`` cannot steal another
    source's values.
    """

    # New producers qualify every colliding/cross-colliding source.  If the
    # complete channel-loss fragment set is present, that mapping is exact and
    # globally injective; prefer it over the compatibility heuristics below.
    if all(fragment in available_suffixes for fragment in fragments):
        return {fragment: fragment for fragment in fragments}

    friendly_candidates: dict[str, list[str]] = {}
    candidate_owners: dict[str, set[str]] = {}
    for fragment in fragments:
        stable_key, label = descriptors[fragment]
        stable_prefix = f"{stable_key}__"
        display_fragment = fragment[len(stable_prefix) :] if fragment.startswith(stable_prefix) else ""
        candidates = list(dict.fromkeys((display_fragment, _sanitized_metric_fragment(label))))
        friendly_candidates[fragment] = [candidate for candidate in candidates if candidate]
        for candidate in friendly_candidates[fragment]:
            candidate_owners.setdefault(candidate, set()).add(fragment)

    resolved: dict[str, str] = {}
    used_suffixes: set[str] = set()
    for fragment in fragments:
        for candidate in friendly_candidates[fragment]:
            if (
                candidate in available_suffixes
                and candidate not in used_suffixes
                and len(candidate_owners[candidate]) == 1
            ):
                resolved[fragment] = candidate
                used_suffixes.add(candidate)
                break

    for fragment in fragments:
        if fragment not in resolved and fragment in available_suffixes and fragment not in used_suffixes:
            resolved[fragment] = fragment
            used_suffixes.add(fragment)
    return resolved


def _select_representative_points(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    if max_points == 1:
        return [points[-1]]

    last_index = len(points) - 1
    last_index_by_source = {}
    for index, point in enumerate(points):
        for source_key in point["values"]:
            last_index_by_source[source_key] = index

    mandatory_indices = {0, last_index, *last_index_by_source.values()}
    if len(mandatory_indices) > max_points:
        prioritized_indices = [last_index, *sorted(last_index_by_source.values(), reverse=True), 0]
        mandatory_indices = set(list(dict.fromkeys(prioritized_indices))[:max_points])

    indices = set(mandatory_indices)
    for candidate_index in (round(index * last_index / (max_points - 1)) for index in range(max_points)):
        if len(indices) >= max_points:
            break
        indices.add(candidate_index)
    return [points[index] for index in sorted(indices)]


@dataclass
class ChannelLossDashboardData:
    """Bounded-render, in-memory view of sampled channel-loss steps."""

    points: list[dict[str, Any]] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    sampled_points: int = 0

    def record(
        self,
        trainer: BaseTrainer,
        step: int,
        metrics: Mapping[str, Any],
        trace_step_id: Any = None,
    ) -> bool:
        callback = getattr(trainer, "channel_loss_callback", None)
        config = getattr(callback, "config", None)
        loss_prefix = getattr(config, "loss_metric_prefix", "channel_loss")
        weighted_prefix = getattr(config, "weighted_loss_metric_prefix", "channel_loss_weighted")
        token_prefix = getattr(config, "token_count_metric_prefix", "channel_tokens")

        raw_values = _metric_values(metrics, loss_prefix)
        if not raw_values:
            return False

        weighted_values = _metric_values(metrics, weighted_prefix)
        token_values = _metric_values(metrics, token_prefix)
        sample_values = _metric_values(metrics, "samples")
        input_token_values = _metric_values(metrics, "input_tokens")
        label_token_values = _metric_values(metrics, "label_tokens")
        label_tokens_per_sample_values = _metric_values(metrics, "label_tokens_per_sample")
        fragments = sorted(raw_values)
        descriptors = _source_descriptors(trainer, fragments)
        data_suffixes = _data_metric_suffixes(
            set(sample_values)
            | set(input_token_values)
            | set(label_token_values)
            | set(label_tokens_per_sample_values),
            fragments,
            descriptors,
        )

        if weighted_values and all(fragment in weighted_values for fragment in fragments):
            overall_text_loss = sum(weighted_values[fragment] for fragment in fragments)
        else:
            overall_text_loss = _finite_float(metrics.get(_FOUNDATION_LOSS_KEY))

        values = {}
        for fragment in fragments:
            stable_key, label = descriptors[fragment]
            data_suffix = data_suffixes.get(fragment)
            self.labels[stable_key] = label
            values[stable_key] = {
                "raw": raw_values[fragment],
                "weighted": weighted_values.get(fragment),
                "tokens": token_values.get(fragment),
                "samples": sample_values.get(data_suffix) if data_suffix is not None else None,
                "input_tokens": input_token_values.get(data_suffix) if data_suffix is not None else None,
                "label_tokens": label_token_values.get(data_suffix) if data_suffix is not None else None,
                "label_tokens_per_sample": (
                    label_tokens_per_sample_values.get(data_suffix) if data_suffix is not None else None
                ),
            }
        # Optional adapter contract: dataloaders with offline sample tracing may
        # publish an opaque step identifier without exposing sample content to
        # the framework dashboard.
        point = {
            "step": int(step),
            "overall": overall_text_loss,
            "trace_step_id": str(trace_step_id) if trace_step_id is not None else None,
            "values": values,
        }
        if self.points and self.points[-1]["step"] == point["step"]:
            self.points[-1] = point
        else:
            self.points.append(point)
            self.sampled_points += 1
        if len(self.points) > _MAX_RETAINED_POINTS:
            self.points = _select_representative_points(self.points, (_MAX_RETAINED_POINTS + 1) // 2)
            retained_sources = {
                source_key for retained_point in self.points for source_key in retained_point["values"]
            }
            self.labels = {
                source_key: label for source_key, label in self.labels.items() if source_key in retained_sources
            }
        return True

    def _render_points(self, max_points: int) -> list[dict[str, Any]]:
        return _select_representative_points(self.points, max_points)

    def payload(self, max_points: int) -> dict[str, Any]:
        points = self._render_points(max_points)
        present_fragments = {fragment for point in points for fragment in point["values"]}
        sources = [
            {"key": fragment, "label": self.labels.get(fragment, fragment)} for fragment in sorted(present_fragments)
        ]
        return {
            "points": points,
            "sources": sources,
            "sampled_points": self.sampled_points,
            "retained_points": len(self.points),
            "rendered_points": len(points),
            "has_weighted": any(
                source_values.get("weighted") is not None
                for point in points
                for source_values in point["values"].values()
            ),
            "has_tokens": any(
                source_values.get("label_tokens") is not None or source_values.get("tokens") is not None
                for point in points
                for source_values in point["values"].values()
            ),
            "has_data_metrics": any(
                source_values.get("samples") is not None
                for point in points
                for source_values in point["values"].values()
            ),
        }

    def render_html(self, max_points: int = _DEFAULT_MAX_RENDER_POINTS) -> str:
        payload = json.dumps(self.payload(max_points), ensure_ascii=False, separators=(",", ":"))
        # Prevent source-controlled strings from terminating the data script.
        payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        return _HTML_TEMPLATE.replace("__CHANNEL_LOSS_PAYLOAD__", payload)


class ChannelLossDashboardCallback(Callback):
    """Publish a compact, interactive dashboard without affecting training."""

    def __init__(self, trainer: BaseTrainer) -> None:
        super().__init__(trainer)
        args = trainer.args
        channel_config = getattr(args.train, "channel_loss", None)
        self.enabled = bool(
            getattr(channel_config, "enable", False) and args.train.global_rank == 0 and args.train.wandb.enable
        )
        self.data = ChannelLossDashboardData()
        self.max_render_points = _positive_int_env("VEOMNI_CHANNEL_DASHBOARD_MAX_POINTS", _DEFAULT_MAX_RENDER_POINTS)
        self._last_published_step: int | None = None
        self._latest_completed_step: int | None = None
        self._exit_hook_registered = False

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self._latest_completed_step = int(state.global_step)
        pop_trace_step_id = getattr(self.trainer, "pop_channel_loss_trace_step_id", None)
        trace_step_id = pop_trace_step_id() if callable(pop_trace_step_id) else None
        metrics = getattr(self.trainer, "step_env_metrics", None)
        if not isinstance(metrics, Mapping):
            return
        try:
            recorded = self.data.record(self.trainer, state.global_step, metrics, trace_step_id=trace_step_id)
        except Exception as exc:  # observability must not take down training
            logger.warning_once(f"Channel-loss dashboard collection failed and was skipped: {exc}")
            return
        if recorded and not self._exit_hook_registered:
            # W&B initializes during on_train_begin. Registering here makes this
            # hook run before W&B's earlier shutdown hook (atexit is LIFO).
            atexit.register(self._publish_latest_at_exit)
            self._exit_hook_registered = True
        # Native scalars remain live in W&B.  Do not upload partial HTML
        # snapshots: W&B may select the earliest media step by default, which
        # made a long run look like it only contained step 1.
        return

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        if self.enabled and self.data.points:
            self._publish(state.global_step)
        if self._exit_hook_registered:
            atexit.unregister(self._publish_latest_at_exit)
            self._exit_hook_registered = False

    def _publish_latest_at_exit(self) -> None:
        """Best-effort flush for uncaught exceptions; SIGKILL remains unrecoverable."""

        if (
            self.enabled
            and self.data.points
            and self._latest_completed_step is not None
            and self._last_published_step != self._latest_completed_step
        ):
            self._publish(self._latest_completed_step)

    def _publish(self, step: int) -> None:
        try:
            import wandb

            if wandb.run is None:
                return
            dashboard = wandb.Html(self.data.render_html(self.max_render_points), inject=False)
            wandb.log({_DASHBOARD_KEY: dashboard}, step=step)
            self._last_published_step = int(step)
        except Exception as exc:  # observability must not take down training
            logger.warning_once(f"Channel-loss dashboard upload failed and was skipped: {exc}")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light;--ink:#172033;--muted:#6a7280;--grid:#e8ebf0;--border:#dfe3e8;--panel:#fff;--bg:#f6f7f9;--accent:#654ff0}
*{box-sizing:border-box}html,body{max-width:100%}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.dashboard{width:100%;min-width:0;padding:14px}.topbar{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin:0 2px 12px}.title{font-size:18px;font-weight:720;letter-spacing:-.02em}.subtitle{color:var(--muted);margin-top:2px}.stats{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.pill{background:#eef0f4;border-radius:999px;padding:5px 9px;color:#4b5360;white-space:nowrap}.panel{position:relative;min-width:0;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 14px 10px;box-shadow:0 1px 2px rgba(16,24,40,.035);margin-bottom:12px}.panel-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:5px}.panel-title{font-weight:680;font-size:14px}.panel-note{color:var(--muted);font-size:12px;margin-top:1px}.toggle{display:inline-flex;flex:none;border:1px solid var(--border);border-radius:8px;padding:2px;background:#f5f6f8}.toggle button{border:0;background:transparent;color:#5c6470;border-radius:6px;padding:5px 10px;cursor:pointer}.toggle button.active{background:#fff;color:var(--ink);box-shadow:0 1px 3px rgba(16,24,40,.14);font-weight:650}.toggle button:disabled{opacity:.4;cursor:not-allowed}.step-select{border:1px solid var(--border);border-radius:8px;background:#fff;color:var(--ink);padding:6px 28px 6px 9px;font:inherit}.chart{width:100%;min-height:285px;display:block;overflow:visible}.legend{display:flex;gap:7px 12px;flex-wrap:wrap;padding:3px 42px 0 58px;max-height:58px;overflow:auto}.legend button{display:inline-flex;align-items:center;gap:6px;border:0;background:transparent;color:#3f4752;padding:1px 0;cursor:pointer;font-size:12px}.legend button.off{opacity:.35;text-decoration:line-through}.swatch{width:9px;height:9px;border-radius:50%;flex:none}.foundation-swatch{width:14px;height:0;border-top:2px dashed #202631;border-radius:0}.tooltip{position:absolute;z-index:5;pointer-events:none;display:none;min-width:min(190px,calc(100% - 16px));max-width:min(310px,calc(100% - 16px));background:rgba(24,31,43,.96);color:#fff;border-radius:8px;padding:8px 10px;box-shadow:0 5px 18px rgba(0,0,0,.22);font-size:12px}.tooltip .step{font-weight:700;margin-bottom:4px}.tooltip .row{display:flex;justify-content:space-between;gap:16px}.tooltip .name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.tooltip .value{font-variant-numeric:tabular-nums}.table-wrap{overflow:auto;margin-top:9px;border:1px solid var(--border);border-radius:9px}.summary-table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}.summary-table th,.summary-table td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--grid)}.summary-table th{background:#f7f8fa;color:#5b6470;font-size:11px;font-weight:650}.summary-table th:first-child,.summary-table td:first-child{text-align:left}.summary-table tbody tr:last-child td{border-bottom:0}.summary-table .total td{font-weight:700;background:#fafbfc}.trace-note{color:var(--muted);font-size:11px;margin:8px 2px 1px;overflow-wrap:anywhere}.trace-note code{color:#4b5360}.footer{color:var(--muted);font-size:11px;padding:0 3px 2px}.empty{padding:80px 20px;text-align:center;color:var(--muted)}
@media(max-width:520px){
  body{font-size:11px}.dashboard{padding:7px}.topbar{display:block;margin:0 1px 7px}.title{font-size:14px}.subtitle,.stats,.panel-note,.footer{display:none}.panel{border-radius:8px;padding:8px 7px 6px;margin-bottom:7px}.panel-head{display:block;margin-bottom:1px}.panel-title{font-size:12px}.toggle{display:flex;width:100%;margin-top:5px}.toggle button{flex:1;min-width:0;padding:3px 4px;font-size:10px;white-space:nowrap}.step-select{width:100%;margin-top:5px;padding:4px 7px}.chart{height:132px;min-height:132px}.legend{gap:2px 8px;padding:1px 5px 0;max-height:32px}.legend button{font-size:10px;max-width:46%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.swatch{width:7px;height:7px}.foundation-swatch{width:10px}.tooltip{font-size:10px;min-width:min(150px,calc(100% - 16px));padding:6px 7px}.summary-table th,.summary-table td{padding:5px 7px}.trace-note{font-size:9px}
}
</style>
</head>
<body>
<div class="dashboard">
  <div class="topbar">
    <div><div class="title">Channel loss · source overview</div><div class="subtitle">One shared axis per question; click a source label to isolate noise.</div></div>
    <div class="stats" id="stats"></div>
  </div>
  <section class="panel" id="loss-panel">
    <div class="panel-head">
      <div><div class="panel-title">Source learning</div><div class="panel-note" id="loss-note">Average CE per supervised token; lower is better.</div></div>
      <div class="toggle"><button id="raw-button" class="active">Raw CE</button><button id="weighted-button">Weighted contribution</button></div>
    </div>
    <svg class="chart" id="loss-chart" viewBox="0 0 960 300" preserveAspectRatio="none"></svg>
    <div class="legend" id="loss-legend"></div><div class="tooltip" id="loss-tooltip"></div>
  </section>
  <section class="panel" id="mix-panel">
    <div class="panel-head"><div><div class="panel-title">Actual data mix</div><div class="panel-note">Share of supervised label tokens observed by channel loss, stacked to 100%.</div></div></div>
    <svg class="chart" id="mix-chart" viewBox="0 0 960 300" preserveAspectRatio="none"></svg>
    <div class="legend" id="mix-legend"></div><div class="tooltip" id="mix-tooltip"></div>
  </section>
  <section class="panel" id="summary-panel">
    <div class="panel-head">
      <div><div class="panel-title">Step sample summary</div><div class="panel-note">Click either chart or choose a step. Counts are globally reduced by the channel-loss callback.</div></div>
      <select class="step-select" id="summary-step" aria-label="Summary optimizer step"></select>
    </div>
    <div class="table-wrap"><table class="summary-table"><thead><tr><th>Source</th><th>Samples</th><th>Input tokens</th><th>Label tokens</th><th>Label / sample</th><th>Trace</th></tr></thead><tbody id="summary-body"></tbody></table></div>
    <div class="trace-note" id="trace-note"></div>
  </section>
  <div class="footer">Native channel-loss scalars remain available for live W&amp;B comparison. Sample-level details remain in optional dataloader-owned trace artifacts; this dashboard renders aggregate metrics only.</div>
</div>
<script>
const DATA=__CHANNEL_LOSS_PAYLOAD__;
const NS="http://www.w3.org/2000/svg";
const COLORS=["#654ff0","#10a37f","#ef7d32","#2d7ff9","#d84f8b","#00a6b2","#9b6b35","#7a63a8","#cf4b42","#5b8c3a","#64748b","#e2a400"];
let mode="raw",summaryStep=DATA.points.at(-1)?.step??null;const hidden=new Set();
const sourceMap=new Map(DATA.sources.map((s,i)=>[s.key,{...s,color:COLORS[i%COLORS.length]}]));
const SHARES=DATA.points.map(p=>{let total=0;const values={};for(const src of DATA.sources){const item=p.values[src.key],value=item?.label_tokens??item?.tokens??0;values[src.key]=value;total+=value}return Object.fromEntries(DATA.sources.map(src=>[src.key,total>0?values[src.key]/total:0]))});
const fmt=v=>v==null?"—":(Math.abs(v)>=1000?v.toLocaleString(undefined,{maximumFractionDigits:0}):v.toLocaleString(undefined,{maximumFractionDigits:4}));
const pct=v=>v==null?"—":(100*v).toFixed(v<.01?2:1)+"%";
const node=(name,attrs={},text=null)=>{const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==null)n.textContent=text;return n};
const visibleSources=()=>DATA.sources.filter(s=>!hidden.has(s.key));
function scales(points,values,yFloor=0,yCeil=null){const W=960,H=300,m={l:58,r:18,t:14,b:38};let xmin=Infinity,xmax=-Infinity,lo=Infinity,hi=-Infinity;for(const p of points){xmin=Math.min(xmin,p.step);xmax=Math.max(xmax,p.step)}for(const v of values){if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v)}}if(!Number.isFinite(lo)||!Number.isFinite(hi)){lo=0;hi=1}lo=Math.min(lo,yFloor);hi=yCeil??hi;if(hi<=lo)hi=lo+1;hi*=yCeil==null?1.06:1;const x=v=>m.l+(v-xmin)/Math.max(1,xmax-xmin)*(W-m.l-m.r);const y=v=>m.t+(hi-v)/(hi-lo)*(H-m.t-m.b);return{W,H,m,x,y,lo,hi,xmin,xmax}}
function axes(svg,s,yTicks=5,percent=false){svg.append(node("line",{x1:s.m.l,y1:s.m.t,x2:s.m.l,y2:s.H-s.m.b,stroke:"#aeb5bf"}));svg.append(node("line",{x1:s.m.l,y1:s.H-s.m.b,x2:s.W-s.m.r,y2:s.H-s.m.b,stroke:"#aeb5bf"}));for(let i=0;i<=yTicks;i++){const v=s.lo+(s.hi-s.lo)*i/yTicks,y=s.y(v);svg.append(node("line",{x1:s.m.l,y1:y,x2:s.W-s.m.r,y2:y,stroke:"#e8ebf0"}));svg.append(node("text",{x:s.m.l-9,y:y+4,"text-anchor":"end",fill:"#737b88","font-size":"11"},percent?Math.round(v*100)+"%":fmt(v)))}const ticks=Math.min(6,DATA.points.length);for(let i=0;i<ticks;i++){const v=s.xmin+(s.xmax-s.xmin)*i/Math.max(1,ticks-1),x=s.x(v);svg.append(node("text",{x,y:s.H-13,"text-anchor":"middle",fill:"#737b88","font-size":"11"},fmt(Math.round(v))))}svg.append(node("text",{x:(s.m.l+s.W-s.m.r)/2,y:s.H-1,"text-anchor":"middle",fill:"#737b88","font-size":"11"},"optimizer step"))}
function pathFor(points,s,getter){let d="",started=false;for(const p of points){const v=getter(p);if(v==null){started=false;continue}d+=(started?"L":"M")+s.x(p.step).toFixed(2)+","+s.y(v).toFixed(2);started=true}return d}
function sampledMarkers(points,max=200){if(points.length<=max)return points;return Array.from({length:max},(_,i)=>points[Math.round(i*(points.length-1)/(max-1))])}
function renderLegend(id){const el=document.getElementById(id);el.replaceChildren();for(const src of DATA.sources){const b=document.createElement("button");b.className=hidden.has(src.key)?"off":"";const sw=document.createElement("span");sw.className="swatch";sw.style.background=sourceMap.get(src.key).color;b.append(sw,document.createTextNode(src.label));b.onclick=()=>{hidden.has(src.key)?hidden.delete(src.key):hidden.add(src.key);renderAll()};el.append(b)}if(id==="loss-legend"){const b=document.createElement("button"),sw=document.createElement("span");sw.className="foundation-swatch";b.append(sw,document.createTextNode("overall text loss"));el.append(b)}}
function installHover(svg,panel,tooltip,s,rowsFor){const line=node("line",{y1:s.m.t,y2:s.H-s.m.b,stroke:"#9299a5","stroke-dasharray":"3 3",display:"none"});svg.append(line);const overlay=node("rect",{x:s.m.l,y:s.m.t,width:s.W-s.m.l-s.m.r,height:s.H-s.m.t-s.m.b,fill:"transparent",cursor:"crosshair"});svg.append(overlay);const nearest=e=>{const r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)/r.width*960,step=s.xmin+(vx-s.m.l)/(s.W-s.m.l-s.m.r)*(s.xmax-s.xmin);let pointIndex=0;for(let i=1;i<DATA.points.length;i++){if(Math.abs(DATA.points[i].step-step)<Math.abs(DATA.points[pointIndex].step-step))pointIndex=i}return[DATA.points[pointIndex],pointIndex]};overlay.addEventListener("mouseleave",()=>{line.setAttribute("display","none");tooltip.style.display="none"});overlay.addEventListener("click",e=>setSummaryStep(nearest(e)[0].step));overlay.addEventListener("mousemove",e=>{const [p,pointIndex]=nearest(e);line.setAttribute("x1",s.x(p.step));line.setAttribute("x2",s.x(p.step));line.setAttribute("display","block");const rows=rowsFor(p,pointIndex);tooltip.innerHTML="";const head=document.createElement("div");head.className="step";head.textContent="step "+p.step;tooltip.append(head);for(const [name,value] of rows){const row=document.createElement("div"),n=document.createElement("span"),v=document.createElement("span");row.className="row";n.className="name";v.className="value";n.textContent=name;v.textContent=value;row.append(n,v);tooltip.append(row)}tooltip.style.display="block";const pr=panel.getBoundingClientRect();let left=e.clientX-pr.left+12;if(left+tooltip.offsetWidth>pr.width-8)left=e.clientX-pr.left-tooltip.offsetWidth-12;left=Math.max(8,Math.min(left,Math.max(8,pr.width-tooltip.offsetWidth-8)));tooltip.style.left=left+"px";tooltip.style.top=(e.clientY-pr.top+10)+"px"})}
function renderLoss(){const svg=document.getElementById("loss-chart");svg.replaceChildren();if(!DATA.points.length){svg.replaceWith(Object.assign(document.createElement("div"),{className:"empty",textContent:"No channel-loss samples yet."}));return}const vals=[];for(const p of DATA.points){if(p.overall!=null)vals.push(p.overall);for(const src of visibleSources()){const v=p.values[src.key]?.[mode];if(v!=null)vals.push(v)}}const s=scales(DATA.points,vals,0);axes(svg,s);for(const src of visibleSources()){const observations=DATA.points.filter(p=>p.values[src.key]?.[mode]!=null),color=sourceMap.get(src.key).color,d=pathFor(DATA.points,s,p=>p.values[src.key]?.[mode]??null);if(d)svg.append(node("path",{d,fill:"none",stroke:color,"stroke-width":"2.2","vector-effect":"non-scaling-stroke"}));for(const p of sampledMarkers(observations))svg.append(node("circle",{cx:s.x(p.step),cy:s.y(p.values[src.key][mode]),r:"2.5",fill:color,"vector-effect":"non-scaling-stroke"}))}const overallPoints=DATA.points.filter(p=>p.overall!=null),overallPath=pathFor(DATA.points,s,p=>p.overall);if(overallPath)svg.append(node("path",{d:overallPath,fill:"none",stroke:"#202631","stroke-width":"1.8","stroke-dasharray":"6 4","vector-effect":"non-scaling-stroke"}));for(const p of sampledMarkers(overallPoints))svg.append(node("circle",{cx:s.x(p.step),cy:s.y(p.overall),r:"2.1",fill:"#202631","vector-effect":"non-scaling-stroke"}));installHover(svg,document.getElementById("loss-panel"),document.getElementById("loss-tooltip"),s,p=>{const rows=visibleSources().map(src=>[src.label,fmt(p.values[src.key]?.[mode])]);rows.push(["overall text loss",fmt(p.overall)]);return rows});renderLegend("loss-legend")}
function renderMix(){const svg=document.getElementById("mix-chart");svg.replaceChildren();if(!DATA.points.length)return;const s=scales(DATA.points,[0,1],0,1);s.hi=1;s.lo=0;s.y=v=>s.m.t+(1-v)*(s.H-s.m.t-s.m.b);axes(svg,s,4,true);if(DATA.points.length===1){let cumulative=0;for(const src of DATA.sources){const share=SHARES[0][src.key]??0,bottom=cumulative;cumulative+=share;if(hidden.has(src.key))continue;svg.append(node("rect",{x:s.m.l,y:s.y(cumulative),width:s.W-s.m.l-s.m.r,height:s.y(bottom)-s.y(cumulative),fill:sourceMap.get(src.key).color,"fill-opacity":".72",stroke:"#fff","stroke-width":".7"}))}}else{const cumulative=DATA.points.map(()=>0);for(const src of DATA.sources){if(hidden.has(src.key))continue;const top=[],bottom=[];DATA.points.forEach((p,i)=>{bottom.push(cumulative[i]);cumulative[i]+=SHARES[i][src.key]??0;top.push(cumulative[i])});const upper=DATA.points.map((p,i)=>s.x(p.step)+","+s.y(top[i])).join(" L");const lower=[...DATA.points].reverse().map((p,j)=>{const i=DATA.points.length-1-j;return s.x(p.step)+","+s.y(bottom[i])}).join(" L");svg.append(node("path",{d:"M"+upper+" L"+lower+" Z",fill:sourceMap.get(src.key).color,"fill-opacity":".72",stroke:"#fff","stroke-width":".7","vector-effect":"non-scaling-stroke"}))}}installHover(svg,document.getElementById("mix-panel"),document.getElementById("mix-tooltip"),s,(p,i)=>visibleSources().map(src=>[src.label,pct(SHARES[i][src.key])]));renderLegend("mix-legend")}
function setSummaryStep(step){summaryStep=step;const select=document.getElementById("summary-step");if(select)select.value=String(step);renderSummary()}
function addCell(row,value,className=""){const cell=document.createElement("td");cell.textContent=value;if(className)cell.className=className;row.append(cell)}
function renderSummary(){const panel=document.getElementById("summary-panel");if(!DATA.has_data_metrics||!DATA.points.length){panel.hidden=true;return}panel.hidden=false;const select=document.getElementById("summary-step");if(!select.options.length){for(const point of DATA.points){const option=document.createElement("option");option.value=String(point.step);option.textContent="step "+point.step;select.append(option)}select.onchange=()=>setSummaryStep(Number(select.value))}if(summaryStep==null||!DATA.points.some(point=>point.step===summaryStep))summaryStep=DATA.points.at(-1).step;select.value=String(summaryStep);const point=DATA.points.find(item=>item.step===summaryStep),body=document.getElementById("summary-body");body.replaceChildren();const hasTrace=point.trace_step_id!=null;let totalSamples=0,totalInput=0,totalLabels=0;for(const src of DATA.sources){const values=point.values[src.key];if(!values||values.samples==null)continue;const row=document.createElement("tr");addCell(row,src.label);addCell(row,fmt(values.samples));addCell(row,fmt(values.input_tokens));addCell(row,fmt(values.label_tokens));addCell(row,fmt(values.label_tokens_per_sample));addCell(row,hasTrace?"linked":"aggregate only");body.append(row);totalSamples+=values.samples??0;totalInput+=values.input_tokens??0;totalLabels+=values.label_tokens??0}const total=document.createElement("tr");total.className="total";addCell(total,"TOTAL");addCell(total,fmt(totalSamples));addCell(total,fmt(totalInput));addCell(total,fmt(totalLabels));addCell(total,totalSamples?fmt(totalLabels/totalSamples):"—");addCell(total,hasTrace?"linked":"aggregate only");body.append(total);const note=document.getElementById("trace-note");note.replaceChildren();if(hasTrace){note.append(document.createTextNode("Trace step: "));const code=document.createElement("code");code.textContent=point.trace_step_id;note.append(code,document.createTextNode(" · Exact sample details stay in adapter-owned trace artifacts."))}else{note.textContent="Sample details are unavailable in this panel; enable a compatible dataloader trace adapter for offline drilldown."}}
function renderStats(){const last=DATA.points.at(-1);const stats=document.getElementById("stats");stats.replaceChildren();const detail=DATA.retained_points<DATA.sampled_points?" · "+DATA.retained_points+" retained":DATA.rendered_points<DATA.retained_points?" · "+DATA.rendered_points+" shown":"";for(const text of [last?"latest step "+last.step:"no data",DATA.sources.length+" sources",DATA.sampled_points+" sampled points"+detail]){const p=document.createElement("span");p.className="pill";p.textContent=text;stats.append(p)}}
function renderAll(){renderStats();renderLoss();if(DATA.has_tokens)renderMix();renderSummary()}
document.getElementById("raw-button").onclick=()=>{mode="raw";document.getElementById("raw-button").classList.add("active");document.getElementById("weighted-button").classList.remove("active");document.getElementById("loss-note").textContent="Average CE per supervised token; lower is better.";renderLoss()};
document.getElementById("weighted-button").onclick=()=>{if(!DATA.has_weighted)return;mode="weighted";document.getElementById("weighted-button").classList.add("active");document.getElementById("raw-button").classList.remove("active");document.getElementById("loss-note").textContent="Per-source contribution; contributions sum to the overall supervised text loss.";renderLoss()};
if(!DATA.has_weighted){document.getElementById("weighted-button").disabled=true;document.getElementById("weighted-button").title="channel_loss.log_weighted_loss is disabled"}
if(!DATA.has_tokens)document.getElementById("mix-panel").hidden=true;
renderAll();
</script>
</body>
</html>"""
