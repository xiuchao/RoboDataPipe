from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from quality.results import QualityReport
from quality.review import quality_report_to_dataframe


LEARNING_METRICS = (
    ("Translation smoothness", "translation_smoothness"),
    ("Joint smoothness", "joint_smoothness"),
    ("Trajectory efficiency", "trajectory_efficiency"),
    ("Hesitation fraction", "hesitation_fraction"),
)

DIAGNOSTIC_METRICS = (
    ("Joint path length", "joint_path_length"),
    ("Translation path length", "translation_path_length"),
    ("Rotation path length", "rotation_path_length"),
    ("Angular speed mean", "angular_speed_mean"),
    ("Angular acceleration RMS", "angular_acceleration_rms"),
    ("Timestamp jitter ratio", "jitter_ratio"),
)


def write_quality_dashboard(
    report: QualityReport,
    dataset_name: str,
    output_path: str | Path,
    *,
    sample: int = 0,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        quality_report_to_html(report, dataset_name, sample=sample),
        encoding="utf-8",
    )


def quality_report_to_html(
    report: QualityReport,
    dataset_name: str,
    *,
    sample: int = 0,
) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "HTML quality dashboards require Plotly: python -m pip install plotly"
        ) from error

    frame = quality_report_to_dataframe(report)
    episode_ids = frame["episode_id"].tolist()
    display_episode_ids = [_display_episode_id(episode_id) for episode_id in episode_ids]
    active_metric_columns = _active_metric_columns(frame)
    valid_count = sum(episode.validity for episode in report.per_episode)
    scored_count = int(frame["overall_score"].notna().sum())
    flagged_ids = {
        episode_id
        for episode_ids_for_flag in report.flagged_episodes.values()
        for episode_id in episode_ids_for_flag
    }
    scope = (
        f"First {report.num_episodes} episodes (sample)"
        if sample > 0
        else f"All {report.num_episodes} episodes"
    )

    overview = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Score distribution", "Score by episode"),
        horizontal_spacing=0.12,
    )
    scores = frame["overall_score"].dropna()
    overview.add_trace(
        go.Histogram(x=scores, marker_color="#287271", opacity=0.9, showlegend=False),
        row=1,
        col=1,
    )
    score_colors = [
        "#c84b31" if episode_id in flagged_ids else "#287271"
        for episode_id in episode_ids
    ]
    overview.add_trace(
        go.Scatter(
            x=display_episode_ids,
            y=frame["overall_score"],
            mode="markers+lines+text",
            marker={"color": score_colors, "size": 8},
            line={"color": "#9aa5a1", "width": 1},
            text=_tail_labels(frame["overall_score"], display_episode_ids),
            textposition="top center",
            customdata=frame["flags"],
            hovertemplate="%{x}<br>Score: %{y:.2f}<br>Flags: %{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    if len(scores):
        overview.add_hline(
            y=float(scores.median()),
            line_dash="dash",
            line_color="#d99b45",
            annotation_text="median",
            row=1,
            col=2,
        )
    overview.update_layout(height=390, margin={"l": 45, "r": 20, "t": 55, "b": 80})
    overview.update_xaxes(tickangle=-45, row=1, col=2)

    learning = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(label for label, _ in LEARNING_METRICS),
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )
    for index, (label, metric_name) in enumerate(LEARNING_METRICS):
        row, column = divmod(index, 2)
        values = _metric_series(frame, active_metric_columns, metric_name, "min")
        if metric_name == "hesitation_fraction":
            values = _metric_series(frame, active_metric_columns, metric_name, "max")
        learning.add_trace(
            go.Scatter(
                x=display_episode_ids,
                y=values,
                mode="markers+lines+text",
                marker={"color": "#287271", "size": 7},
                line={"color": "#b4bfbb", "width": 1},
                text=_tail_labels(values, display_episode_ids),
                textposition="top center",
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.3f}}<extra></extra>",
                showlegend=False,
            ),
            row=row + 1,
            col=column + 1,
        )
        if metric_name == "hesitation_fraction":
            learning.add_hline(
                y=0.2,
                line_dash="dash",
                line_color="#c84b31",
                annotation_text="flag threshold",
                row=row + 1,
                col=column + 1,
            )
    learning.update_layout(height=680, margin={"l": 45, "r": 20, "t": 55, "b": 65})
    learning.update_xaxes(tickangle=-45)

    diagnostics = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=tuple(label for label, _ in DIAGNOSTIC_METRICS),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    for index, (label, metric_name) in enumerate(DIAGNOSTIC_METRICS):
        row, column = divmod(index, 2)
        values = _metric_series(frame, active_metric_columns, metric_name, "max")
        numeric = values.dropna()
        if len(numeric):
            p10, median, p90 = np.percentile(numeric, [10, 50, 90])
            diagnostics.add_hrect(
                y0=p10,
                y1=p90,
                fillcolor="#dce8e4",
                opacity=0.6,
                line_width=0,
                row=row + 1,
                col=column + 1,
            )
            diagnostics.add_hline(
                y=median,
                line_dash="dash",
                line_color="#5e6b66",
                row=row + 1,
                col=column + 1,
            )
        diagnostics.add_trace(
            go.Scatter(
                x=display_episode_ids,
                y=values,
                mode="markers+text",
                marker={"color": "#d99b45", "size": 7},
                text=_tail_labels(values, display_episode_ids),
                textposition="top center",
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.3f}}<extra></extra>",
                showlegend=False,
            ),
            row=row + 1,
            col=column + 1,
        )
    diagnostics.update_layout(height=900, margin={"l": 45, "r": 20, "t": 55, "b": 65})
    diagnostics.update_xaxes(tickangle=-45)

    efficiency = _metric_series(frame, active_metric_columns, "trajectory_efficiency", "min")
    smoothness = _metric_series(frame, active_metric_columns, "translation_smoothness", "min")
    hesitation = _metric_series(frame, active_metric_columns, "hesitation_fraction", "max")
    cross_labels = _combined_tail_labels(
        display_episode_ids,
        efficiency,
        smoothness,
        hesitation,
    )
    cross_metric = go.Figure(
        go.Scatter(
            x=efficiency,
            y=smoothness,
            mode="markers+text",
            text=cross_labels,
            textposition=_inward_label_positions(efficiency, smoothness, cross_labels),
            textfont={"size": 10},
            hovertext=display_episode_ids,
            marker={
                "color": hesitation,
                "colorscale": [[0, "#287271"], [1, "#c84b31"]],
                "colorbar": {"title": "Hesitation"},
                "size": 11,
            },
            customdata=frame["overall_score"],
            hovertemplate=(
                "%{hovertext}<br>Efficiency: %{x:.3f}<br>Smoothness: %{y:.3f}"
                "<br>Score: %{customdata:.2f}<extra></extra>"
            ),
        )
    )
    cross_metric.update_layout(
        height=500,
        xaxis_title="Trajectory efficiency",
        yaxis_title="Translation smoothness",
        margin={"l": 60, "r": 30, "t": 25, "b": 60},
    )

    return _dashboard_document(
        title=f"{dataset_name} Dataset Quality",
        subtitle=f"{scope} | Generated {report.computed_at}",
        cards=(
            ("Dataset score", f"{report.overall_score:.2f}", "/ 10"),
            ("Valid", f"{valid_count}/{report.num_episodes}", "episodes"),
            ("Scored", f"{scored_count}/{report.num_episodes}", "episodes"),
            ("Flagged", str(len(flagged_ids)), "episodes"),
        ),
        overview=_figure_div(overview, include_plotlyjs=True),
        learning=_figure_div(learning),
        diagnostics=_figure_div(diagnostics),
        cross_metric=_figure_div(cross_metric),
        review_table=_review_table(frame, flagged_ids),
        sample=sample,
    )


def _active_metric_columns(frame, metric_name: str | None = None) -> dict[str, list[str]]:
    metric_names = {
        name for _, name in (*LEARNING_METRICS, *DIAGNOSTIC_METRICS)
    }
    if metric_name is not None:
        metric_names = {metric_name}
    return {
        name: [
            column
            for column in frame.columns
            if column.endswith(f"_{name}") and not column.endswith(f"cartesian_{name}")
        ]
        for name in metric_names
    }


def _metric_series(frame, columns: dict[str, list[str]], metric_name: str, mode: str):
    selected = columns.get(metric_name, [])
    if not selected:
        return frame["overall_score"] * np.nan
    if mode == "min":
        return frame[selected].min(axis=1, skipna=True)
    return frame[selected].max(axis=1, skipna=True)


def _tail_labels(values, episode_ids: list[str]) -> list[str]:
    numeric = np.asarray(values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) < 2:
        return ["" for _ in episode_ids]
    p10, p90 = np.percentile(finite, [10, 90])
    return [
        episode_id
        if np.isfinite(value)
        and (
            (value < p10 and not np.isclose(value, p10, atol=0.005))
            or (value > p90 and not np.isclose(value, p90, atol=0.005))
        )
        else ""
        for value, episode_id in zip(numeric, episode_ids)
    ]


def _combined_tail_labels(episode_ids: list[str], *series) -> list[str]:
    tail_sets = [
        set(label for label in _tail_labels(values, episode_ids) if label)
        for values in series
    ]
    outliers = set().union(*tail_sets)
    return [episode_id if episode_id in outliers else "" for episode_id in episode_ids]


def _inward_label_positions(x_values, y_values, labels: list[str]) -> list[str]:
    x_numeric = np.asarray(x_values, dtype=float)
    y_numeric = np.asarray(y_values, dtype=float)
    x_midpoint = float(np.nanmedian(x_numeric))
    y_high = float(np.nanpercentile(y_numeric, 85))
    positions = []
    for x_value, y_value, label in zip(x_numeric, y_numeric, labels):
        if not label:
            positions.append("top center")
        elif y_value >= y_high:
            positions.append("bottom center")
        elif x_value <= x_midpoint:
            positions.append("middle right")
        else:
            positions.append("middle left")
    return positions


def _figure_div(figure, *, include_plotlyjs: bool = False) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotlyjs else False,
        config={"displaylogo": False, "responsive": True},
    )


def _display_episode_id(episode_id: str) -> str:
    prefix = "episode_"
    suffix = episode_id.removeprefix(prefix)
    return f"ep{suffix}" if episode_id.startswith(prefix) and suffix.isdigit() else episode_id


def _review_table(frame, flagged_ids: set[str]) -> str:
    table = frame.copy()
    table["priority"] = table.apply(
        lambda row: (0 if row["episode_id"] in flagged_ids else 1, row["overall_score"]),
        axis=1,
    )
    table = table.sort_values("priority")
    rows = []
    for _, row in table.iterrows():
        episode_id = escape(_display_episode_id(str(row["episode_id"])))
        flags = escape(str(row["flags"] or ""))
        rows.append(
            "<tr>"
            f"<td>{episode_id}</td>"
            f"<td data-value='{float(row['overall_score']):.6f}'>{row['overall_score']:.2f}</td>"
            f"<td>{flags or '-'}</td>"
            f"<td>{int(row['length'])}</td>"
            "<td><select aria-label='Review status'><option>Pending</option>"
            "<option>Accept</option><option>Reject</option><option>Uncertain</option>"
            "</select></td>"
            "</tr>"
        )
    return (
        '<table id="review-table"><thead><tr>'
        '<th data-column="0">Episode</th><th data-column="1">Score</th>'
        '<th data-column="2">Flags</th><th data-column="3">Frames</th>'
        "<th>Review status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _dashboard_document(
    *,
    title: str,
    subtitle: str,
    cards: tuple[tuple[str, str, str], ...],
    overview: str,
    learning: str,
    diagnostics: str,
    cross_metric: str,
    review_table: str,
    sample: int,
) -> str:
    card_html = "".join(
        f'<div class="metric"><span>{escape(label)}</span><strong>{escape(value)}</strong>'
        f"<small>{escape(detail)}</small></div>"
        for label, value, detail in cards
    )
    sample_note = (
        '<p class="notice">This dashboard covers a leading sample, not the full dataset.</p>'
        if sample > 0
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ --ink:#1c2925; --muted:#66736e; --paper:#f4f6f2; --panel:#fff; --line:#d7ded9; --accent:#287271; --warm:#d99b45; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:linear-gradient(135deg,#edf2ed 0%,#f8f5ee 100%); font-family:"IBM Plex Sans","Noto Sans",sans-serif; }}
header {{ padding:34px clamp(20px,5vw,72px) 24px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-family:"IBM Plex Serif",Georgia,serif; font-size:clamp(28px,4vw,48px); letter-spacing:0; }}
header p {{ margin:0; color:var(--muted); }}
main {{ max-width:1500px; margin:auto; padding:24px clamp(14px,3vw,44px) 60px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:20px; }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--accent); padding:16px; min-height:112px; }}
.metric span,.metric small {{ display:block; color:var(--muted); }} .metric strong {{ display:block; font-size:32px; margin:9px 0 2px; }}
section {{ margin:24px 0 0; padding:18px; background:var(--panel); border:1px solid var(--line); }}
h2 {{ margin:0 0 4px; font-size:20px; letter-spacing:0; }} .hint {{ margin:0 0 12px; color:var(--muted); font-size:14px; }}
.notice {{ border-left:4px solid var(--warm); padding:12px 14px; background:#fff8e9; }}
.table-wrap {{ overflow:auto; max-height:560px; }} table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ position:sticky; top:0; cursor:pointer; background:#e8efeb; text-align:left; }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }}
select {{ border:1px solid #aebbb5; background:white; padding:6px 28px 6px 8px; }}
@media (max-width:760px) {{ .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} section {{ padding:10px; }} }}
</style>
</head>
<body>
<header><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></header>
<main>{sample_note}<div class="metrics">{card_html}</div>
<section><h2>Dataset overview</h2><p class="hint">Flagged episodes are shown in red.</p>{overview}</section>
<section><h2>Learning quality</h2><p class="hint">These metrics contribute to filtering and the overall score.</p>{learning}</section>
<section><h2>Diagnostic distributions</h2><p class="hint">Shaded bands show P10-P90. Diagnostics are context, not quality labels.</p>{diagnostics}</section>
<section><h2>Cross-metric view</h2><p class="hint">Color indicates hesitation fraction.</p>{cross_metric}</section>
<section><h2>Review queue</h2><p class="hint">Click column headers to sort. Review selections remain local to this browser tab.</p><div class="table-wrap">{review_table}</div></section>
</main>
<script>
document.querySelectorAll('#review-table th[data-column]').forEach(function(header) {{
  header.addEventListener('click', function() {{
    const body = document.querySelector('#review-table tbody');
    const index = Number(header.dataset.column);
    const ascending = header.dataset.order !== 'asc';
    const rows = Array.from(body.querySelectorAll('tr'));
    rows.sort(function(a,b) {{
      const av = a.children[index].dataset.value || a.children[index].textContent.trim();
      const bv = b.children[index].dataset.value || b.children[index].textContent.trim();
      const an = Number(av), bn = Number(bv);
      const result = Number.isNaN(an) || Number.isNaN(bn) ? av.localeCompare(bv) : an - bn;
      return ascending ? result : -result;
    }});
    rows.forEach(function(row) {{ body.appendChild(row); }});
    header.dataset.order = ascending ? 'asc' : 'desc';
  }});
}});
</script>
</body></html>"""