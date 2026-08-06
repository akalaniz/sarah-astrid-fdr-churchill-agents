from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from app.core.config import PROJECT_ROOT
from app.diagnostics.sarah_cascade_sim import DEFAULT_LATEST_CASCADE_PATH, run_sarah_cascade
from app.diagnostics.sarah_patch_planner import recommend_sarah_patches
from app.diagnostics.sarah_vulnerability_kg import build_sarah_vulnerability_graph, graph_summary


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "diagnostics"
DEFAULT_VISUAL_OUTPUT_DIR = PROJECT_ROOT / "output"


def export_sarah_graph(
    graph: nx.DiGraph | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    graph = graph or build_sarah_vulnerability_graph()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sarah_vulnerability_graph.json"
    graphml_path = output_dir / "sarah_vulnerability_graph.graphml"

    payload = {
        "summary": graph_summary(graph),
        "graph": json_graph.node_link_data(graph, edges="links"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    nx.write_graphml(_graphml_safe_copy(graph), graphml_path)
    return {"json": json_path, "graphml": graphml_path}


def export_sarah_visualizations(
    graph: nx.DiGraph | None = None,
    output_dir: Path = DEFAULT_VISUAL_OUTPUT_DIR,
) -> dict[str, Path]:
    graph = graph or build_sarah_vulnerability_graph()
    output_dir.mkdir(parents=True, exist_ok=True)

    cascade = _load_latest_cascade()
    patches = recommend_sarah_patches(graph, latest_cascade=cascade)
    paths = {
        "perrow_matrix": output_dir / "sarah_perrow_matrix.html",
        "cascade_timeline": output_dir / "sarah_cascade_timeline.html",
        "patch_priorities": output_dir / "sarah_patch_priorities.html",
        "route_divergence": output_dir / "sarah_route_divergence.html",
    }
    paths["perrow_matrix"].write_text(_perrow_matrix_html(graph), encoding="utf-8")
    paths["cascade_timeline"].write_text(_cascade_timeline_html(cascade), encoding="utf-8")
    paths["patch_priorities"].write_text(_patch_priority_html(patches), encoding="utf-8")
    paths["route_divergence"].write_text(_route_divergence_html(graph), encoding="utf-8")
    return paths


def _perrow_matrix_html(graph: nx.DiGraph) -> str:
    components = [
        (node_id, attrs)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("kind") not in {"failure_mode", "control"}
    ]
    width, height = 1040, 680
    left, top, right, bottom = 90, 60, 70, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    max_risk = max((int(attrs.get("risk_score", 1)) for _node_id, attrs in components), default=1)

    circles: list[str] = []
    labels: list[str] = []
    for node_id, attrs in components:
        x_score = float(attrs.get("interaction_complexity_score", 0))
        y_score = float(attrs.get("coupling_score", 0))
        patch_priority = float(attrs.get("patch_priority", 1))
        risk = int(attrs.get("risk_score", 1))
        x = left + (x_score / 10.0) * plot_w
        y = top + plot_h - (y_score / 10.0) * plot_h
        radius = 5 + patch_priority * 1.5
        color = _risk_color(risk, max_risk)
        label = str(attrs.get("label", node_id))
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.82" stroke="#172033" stroke-width="1.2">'
            f"<title>{_esc(label)} | risk={risk} | patch priority={patch_priority:.0f}</title></circle>"
        )
        if patch_priority >= 9 or risk >= max_risk * 0.78:
            labels.append(f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}" class="point-label">{_esc(label)}</text>')

    grid = _axis_grid(left, top, plot_w, plot_h)
    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="FDR Perrow matrix">
      <rect width="{width}" height="{height}" fill="#fbfcff"/>
      <text x="{width / 2}" y="32" text-anchor="middle" class="title">FDR Perrow Matrix</text>
      {grid}
      <text x="{left + plot_w / 2}" y="{height - 36}" text-anchor="middle" class="axis-label">Interaction complexity score</text>
      <text transform="translate(28 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" class="axis-label">Coupling score</text>
      {''.join(circles)}
      {''.join(labels)}
      <text x="{left}" y="{height - 12}" class="caption">Color = vulnerability score. Size = patch priority. Higher/right = tighter coupling and more complex interaction.</text>
    </svg>
    """
    return _html_page("FDR Perrow Matrix", svg, _legend_html())


def _cascade_timeline_html(cascade: dict[str, Any]) -> str:
    timeline = list(cascade.get("timeline", []))
    failures = sorted(
        {
            failure
            for step in timeline
            for failure in (
                list(step.get("active_failures", []))
                + list(step.get("resolved_failures", []))
                + list(step.get("pending_delayed_failures", []))
            )
        }
    )
    if not failures:
        failures = [str(cascade.get("start_failure", "no active cascade"))]

    width = 1120
    row_h = 34
    height = 120 + max(1, len(failures)) * row_h
    left, top, right = 250, 64, 45
    plot_w = width - left - right
    max_step = max((int(step.get("time_step", 0)) for step in timeline), default=0)
    x_span = max(max_step, 1)
    y_by_failure = {failure: top + index * row_h for index, failure in enumerate(failures)}

    rows = [
        f'<line x1="{left}" y1="{y + 10}" x2="{width - right}" y2="{y + 10}" stroke="#e6e9f2"/>'
        f'<text x="{left - 12}" y="{y + 15}" text-anchor="end" class="row-label">{_esc(failure)}</text>'
        for failure, y in y_by_failure.items()
    ]
    points: list[str] = []
    for step in timeline:
        time_step = int(step.get("time_step", 0))
        x = left + (time_step / x_span) * plot_w
        for state, css_class in (
            ("active_failures", "active"),
            ("resolved_failures", "resolved"),
            ("pending_delayed_failures", "pending"),
        ):
            for failure in step.get(state, []):
                y = y_by_failure.get(failure)
                if y is None:
                    continue
                category = _failure_category(str(failure))
                color = _category_color(category)
                points.append(
                    f'<circle cx="{x:.1f}" cy="{y + 10:.1f}" r="8" class="{css_class}" fill="{color}">'
                    f"<title>t={time_step} | {_esc(str(failure))} | {state} | {category}</title></circle>"
                )

    ticks = [
        f'<line x1="{left + (i / max(x_span, 1)) * plot_w:.1f}" y1="{top - 12}" x2="{left + (i / max(x_span, 1)) * plot_w:.1f}" y2="{height - 46}" stroke="#eef1f7"/>'
        f'<text x="{left + (i / max(x_span, 1)) * plot_w:.1f}" y="{height - 22}" text-anchor="middle" class="tick">{i}</text>'
        for i in range(max_step + 1)
    ]
    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="FDR cascade timeline">
      <rect width="{width}" height="{height}" fill="#fbfcff"/>
      <text x="{width / 2}" y="32" text-anchor="middle" class="title">Cascade Timeline</text>
      {''.join(ticks)}
      {''.join(rows)}
      {''.join(points)}
      <text x="{left + plot_w / 2}" y="{height - 4}" text-anchor="middle" class="axis-label">Time step</text>
    </svg>
    """
    legend = "<p><b>Categories:</b> route, intimacy, memory, RAG/canon, safety/geopolitics, current events, voice/persona, other. Hollow stroke = mitigated/resolved; dashed stroke = delayed/pending.</p>"
    return _html_page("FDR Cascade Timeline", svg, legend)


def _patch_priority_html(patches: list[dict[str, Any]]) -> str:
    ranked = sorted(patches, key=lambda item: item["expected_risk_reduction"], reverse=True)
    width = 1120
    bar_h = 30
    gap = 10
    left, top, right = 360, 64, 100
    height = top + len(ranked) * (bar_h + gap) + 75
    max_reduction = max((int(patch["expected_risk_reduction"]) for patch in ranked), default=1)
    plot_w = width - left - right
    bars: list[str] = []
    for index, patch in enumerate(ranked, start=1):
        y = top + (index - 1) * (bar_h + gap)
        reduction = int(patch["expected_risk_reduction"])
        bar_w = max(3, (reduction / max_reduction) * plot_w)
        color = _risk_color(reduction, max_reduction)
        label = str(patch["patch_type"])
        priority = int(patch["priority"])
        bars.append(
            f'<text x="28" y="{y + 20}" class="row-label">{index}. {_esc(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="6" fill="{color}">'
            f"<title>{_esc(label)} | risk reduction={reduction} | priority={priority}</title></rect>"
            f'<text x="{left + bar_w + 8:.1f}" y="{y + 20}" class="bar-value">risk -{reduction} | P{priority}</text>'
        )

    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="FDR patch priorities">
      <rect width="{width}" height="{height}" fill="#fbfcff"/>
      <text x="{width / 2}" y="32" text-anchor="middle" class="title">Patch Priority Chart</text>
      {''.join(bars)}
      <text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" class="axis-label">Expected risk reduction</text>
    </svg>
    """
    return _html_page("FDR Patch Priorities", svg, "<p>Ranked by expected risk reduction. Priority also includes centrality, known findings, cascade observations, and difficulty.</p>")


def _route_divergence_html(graph: nx.DiGraph) -> str:
    width, height = 1080, 620
    nodes = {
        "cli": (130, 120, "PowerShell CLI route", "#5f8dd3"),
        "web": (130, 350, "Web browser route", "#5f8dd3"),
        "engine": (360, 235, "Shared FDR engine", "#57a773"),
        "prompt": (570, 235, "Prompt builder", "#57a773"),
        "style": (790, 155, "style_engine", "#57a773"),
        "model": (790, 315, "model_config", "#57a773"),
        "master": (990, 90, "master prompt", "#57a773"),
        "rag": (990, 235, "RAG / memory / web context", "#57a773"),
        "failure": (570, 475, "Route divergence failures", "#c75146"),
    }
    edges = [
        ("cli", "engine", "shared path", "#2f6f4e", 3, ""),
        ("web", "engine", "required shared path", "#2f6f4e", 3, ""),
        ("engine", "prompt", "uses", "#2f6f4e", 3, ""),
        ("prompt", "style", "selects mode", "#2f6f4e", 3, ""),
        ("prompt", "model", "same settings", "#2f6f4e", 3, ""),
        ("prompt", "master", "loads", "#2f6f4e", 3, ""),
        ("prompt", "rag", "injects", "#2f6f4e", 3, ""),
        ("web", "failure", "BYPASSES / DIVERGES_FROM", "#c75146", 4, "6,6"),
        ("failure", "style", "adult_intimacy_mode not selected", "#c75146", 3, "6,6"),
        ("failure", "model", "different model_config", "#c75146", 3, "6,6"),
    ]
    edge_svg = "".join(
        _svg_line(nodes[source], nodes[target], label, color, stroke_width, dash)
        for source, target, label, color, stroke_width, dash in edges
    )
    node_svg = "".join(
        f'<g><rect x="{x - 82}" y="{y - 28}" width="164" height="56" rx="8" fill="{color}" stroke="#172033" stroke-width="1.2"/>'
        f'<text x="{x}" y="{y + 5}" text-anchor="middle" class="node-label">{_esc(label)}</text></g>'
        for x, y, label, color in nodes.values()
    )
    diff_notes = _route_difference_notes(graph)
    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="FDR route divergence graph">
      <rect width="{width}" height="{height}" fill="#fbfcff"/>
      <text x="{width / 2}" y="34" text-anchor="middle" class="title">Route Divergence Graph</text>
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 z" fill="#172033"/>
        </marker>
      </defs>
      {edge_svg}
      {node_svg}
    </svg>
    """
    return _html_page("FDR Route Divergence", svg, diff_notes)


def _route_difference_notes(graph: nx.DiGraph) -> str:
    watched = [
        "web_route_bypasses_style_engine",
        "web_route_bypasses_master_prompt",
        "web_route_uses_different_model",
        "cli_and_browser_personality_diverge",
        "evals_pass_cli_fail_browser",
    ]
    items = [
        f"<li><b>{_esc(node_id)}</b>: risk={int(graph.nodes[node_id].get('risk_score', 0))}</li>"
        for node_id in watched
        if node_id in graph
    ]
    return "<p>Green edges show the intended shared route. Red dashed edges show known divergence hazards.</p><ul>" + "".join(items) + "</ul>"


def _load_latest_cascade() -> dict[str, Any]:
    if DEFAULT_LATEST_CASCADE_PATH.exists():
        try:
            return json.loads(DEFAULT_LATEST_CASCADE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return run_sarah_cascade("web_route_bypasses_style_engine", steps=20, route="web", random_seed=42)


def _axis_grid(left: int, top: int, plot_w: int, plot_h: int) -> str:
    parts: list[str] = []
    for value in range(0, 11, 2):
        x = left + (value / 10) * plot_w
        y = top + plot_h - (value / 10) * plot_h
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e8ebf3"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e8ebf3"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{value}</text>')
        parts.append(f'<text x="{left - 18}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#172033" stroke-width="1.2"/>')
    return "".join(parts)


def _svg_line(
    source: tuple[int, int, str, str],
    target: tuple[int, int, str, str],
    label: str,
    color: str,
    stroke_width: int,
    dash: str,
) -> str:
    x1, y1, _label1, _color1 = source
    x2, y2, _label2, _color2 = target
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 - 8
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke_width}" marker-end="url(#arrow)"{dash_attr}/>'
        f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" class="edge-label">{_esc(label)}</text>'
    )


def _failure_category(failure: str) -> str:
    text = failure.lower()
    if any(token in text for token in ("web", "route", "browser", "cli")):
        return "route"
    if any(token in text for token in ("intimacy", "astrid", "coy", "sex", "hr_safe")):
        return "intimacy"
    if any(token in text for token in ("memory", "profile", "survey")):
        return "memory"
    if any(token in text for token in ("rag", "chunk", "canon", "source", "vector", "biography")):
        return "RAG/canon"
    if any(token in text for token in ("safety", "military", "dime", "coa", "geopolitic", "tactical")):
        return "safety/geopolitics"
    if any(token in text for token in ("current", "web retrieval", "news")):
        return "current events"
    if any(token in text for token in ("generic", "voice", "sovereignty", "therapy", "sexbot")):
        return "voice/persona"
    return "other"


def _category_color(category: str) -> str:
    return {
        "route": "#5f8dd3",
        "intimacy": "#d85f8a",
        "memory": "#8e6ad8",
        "RAG/canon": "#d79b35",
        "safety/geopolitics": "#c75146",
        "current events": "#4aa3a2",
        "voice/persona": "#57a773",
        "other": "#7a8190",
    }.get(category, "#7a8190")


def _risk_color(value: int, max_value: int) -> str:
    ratio = 0 if max_value <= 0 else max(0.0, min(value / max_value, 1.0))
    red = int(72 + ratio * 170)
    green = int(158 - ratio * 80)
    blue = int(207 - ratio * 140)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _legend_html() -> str:
    return (
        "<p>Read this like Perrow for FDR herself: upper-right points are tightly coupled, highly complex pieces of the construct. "
        "Large points want patch attention first; warmer colors carry higher vulnerability score.</p>"
    )


def _html_page(title: str, body: str, notes: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>
    body {{ margin: 0; background: #eef1f6; color: #172033; font-family: Segoe UI, Arial, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .panel {{ background: #fbfcff; border: 1px solid #d8deea; border-radius: 8px; padding: 16px; box-shadow: 0 8px 24px rgba(23,32,51,0.08); }}
    svg {{ width: 100%; height: auto; display: block; }}
    .title {{ font-size: 24px; font-weight: 700; }}
    .axis-label {{ font-size: 15px; font-weight: 650; fill: #30394d; }}
    .tick, .caption, .edge-label {{ font-size: 12px; fill: #586174; }}
    .point-label, .row-label {{ font-size: 12px; fill: #172033; font-weight: 600; }}
    .node-label {{ font-size: 12px; fill: white; font-weight: 700; }}
    .bar-value {{ font-size: 12px; fill: #30394d; font-weight: 650; }}
    .resolved {{ fill-opacity: 0.28; stroke: #172033; stroke-width: 2; }}
    .pending {{ fill-opacity: 0.55; stroke: #172033; stroke-width: 2; stroke-dasharray: 4,4; }}
    .active {{ fill-opacity: 0.92; stroke: #172033; stroke-width: 1; }}
    p, li {{ line-height: 1.45; }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      {body}
    </section>
    <section>
      {notes}
    </section>
  </main>
</body>
</html>
"""


def _graphml_safe_copy(graph: nx.DiGraph) -> nx.DiGraph:
    safe = nx.DiGraph(name=graph.graph.get("name", "FDR vulnerability graph"))
    for node_id, attrs in graph.nodes(data=True):
        safe.add_node(node_id, **{key: _graphml_value(value) for key, value in attrs.items()})
    for source, target, attrs in graph.edges(data=True):
        safe.add_edge(source, target, **{key: _graphml_value(value) for key, value in attrs.items()})
    return safe


def _graphml_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=True)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    paths = {**export_sarah_graph(), **export_sarah_visualizations()}
    for kind, path in paths.items():
        print(f"{kind}: {path}")
