from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


STATUS_ORDER = ("inside", "uncertain", "not_inside", "unparsed")
STATUS_COLORS = {
    "inside": "#2f7d5b",
    "uncertain": "#d99b45",
    "not_inside": "#c84b31",
    "unparsed": "#7b8581",
}


@dataclass(frozen=True)
class TaskOutcomeRecord:
    episode_id: str
    status: str
    confidence: float | None
    inference_seconds: float | None = None
    keyframe_dir: str | None = None


def records_from_result(result: dict[str, Any]) -> list[TaskOutcomeRecord]:
    records = []
    for episode_id, episode in result.get("episodes", {}).items():
        parsed = episode.get("parsed_answer")
        parsed = parsed if isinstance(parsed, dict) else {}
        status = str(parsed.get("placement_status", "unparsed"))
        if status not in STATUS_ORDER:
            status = "unparsed"
        confidence = parsed.get("confidence")
        records.append(
            TaskOutcomeRecord(
                episode_id=episode_id,
                status=status,
                confidence=float(confidence) if confidence is not None else None,
                inference_seconds=_optional_float(episode.get("inference_seconds")),
                keyframe_dir=episode.get("keyframe_dir"),
            )
        )
    return records


def records_from_summary(text: str) -> list[TaskOutcomeRecord]:
    matches = list(re.finditer(r"^\[episode\]\s+(ep\d+)\s*$", text, re.MULTILINE))
    records = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():stop]
        answer = _parse_answer_object(block)
        status = str(answer.get("placement_status", "unparsed"))
        if status not in STATUS_ORDER:
            status = "unparsed"
        confidence = answer.get("confidence")
        inference_match = re.search(
            r"^\[episode_inference_seconds\]\s+([0-9.]+)\s*$",
            block,
            re.MULTILINE,
        )
        records.append(
            TaskOutcomeRecord(
                episode_id=match.group(1),
                status=status,
                confidence=float(confidence) if confidence is not None else None,
                inference_seconds=(
                    float(inference_match.group(1)) if inference_match else None
                ),
            )
        )
    return records


def task_outcome_dashboard_html(
    records: list[TaskOutcomeRecord],
    dataset_name: str,
    *,
    subtitle: str = "Task outcome evaluation",
) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "Task outcome dashboards require Plotly: python -m pip install plotly"
        ) from error

    counts = Counter(record.status for record in records)
    success_rate = 100.0 * counts["inside"] / len(records) if records else 0.0
    episode_ids = [record.episode_id for record in records]
    status_codes = [[STATUS_ORDER.index(record.status) for record in records]]
    hover = [
        f"{record.episode_id}<br>Status: {record.status}<br>Confidence: {_confidence(record)}"
        for record in records
    ]

    status_figure = go.Figure(
        go.Heatmap(
            z=status_codes,
            x=episode_ids,
            y=["Outcome"],
            text=[hover],
            hovertemplate="%{text}<extra></extra>",
            zmin=0,
            zmax=len(STATUS_ORDER) - 1,
            colorscale=_status_colorscale(),
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    status_figure.update_layout(
        height=180,
        margin={"l": 75, "r": 20, "t": 20, "b": 65},
        xaxis={"tickangle": -45, "dtick": 2},
    )

    confidence_figure = go.Figure()
    for status in STATUS_ORDER:
        selected = [record for record in records if record.status == status]
        if not selected:
            continue
        confidence_figure.add_trace(
            go.Scatter(
                x=[record.episode_id for record in selected],
                y=[record.confidence for record in selected],
                mode="markers",
                name=status.replace("_", " "),
                marker={"color": STATUS_COLORS[status], "size": 9},
                customdata=[record.inference_seconds for record in selected],
                hovertemplate=(
                    "%{x}<br>Status: " + status.replace("_", " ")
                    + "<br>Confidence: %{y:.2f}<br>Inference: %{customdata:.3f}s"
                    + "<extra></extra>"
                ),
            )
        )
    confidence_figure.update_layout(
        height=360,
        margin={"l": 55, "r": 20, "t": 20, "b": 70},
        xaxis={"title": "Episode", "tickangle": -45, "dtick": 2},
        yaxis={"title": "Confidence", "range": [-0.03, 1.03]},
        legend={"orientation": "h", "y": 1.12},
    )

    review_records = [
        record
        for record in records
        if record.status != "inside" or (record.confidence or 0.0) < 0.8
    ]
    return _document(
        title=f"{dataset_name} Task Outcomes",
        subtitle=subtitle,
        cards=(
            ("Success rate", f"{success_rate:.1f}%", "of all episodes"),
            ("Inside", str(counts["inside"]), f"of {len(records)} episodes"),
            ("Uncertain", str(counts["uncertain"]), "manual review"),
            ("Not inside", str(counts["not_inside"]), "task failures"),
        ),
        status_figure=_figure_div(status_figure, include_plotlyjs=True),
        confidence_figure=_figure_div(confidence_figure),
        review_table=_review_table(review_records),
    )


def write_task_outcome_dashboard(
    records: list[TaskOutcomeRecord],
    dataset_name: str,
    output_path: str | Path,
    *,
    subtitle: str = "Task outcome evaluation",
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        task_outcome_dashboard_html(records, dataset_name, subtitle=subtitle),
        encoding="utf-8",
    )


def _parse_answer_object(block: str) -> dict[str, Any]:
    for candidate in re.findall(r"\{.*?\}", block, re.DOTALL):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "placement_status" in parsed:
            return parsed
    return {}


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _confidence(record: TaskOutcomeRecord) -> str:
    return f"{record.confidence:.2f}" if record.confidence is not None else "n/a"


def _status_colorscale() -> list[list[Any]]:
    scale = []
    maximum = len(STATUS_ORDER) - 1
    for index, status in enumerate(STATUS_ORDER):
        start = max(0.0, (index - 0.49) / maximum)
        stop = min(1.0, (index + 0.49) / maximum)
        scale.extend([[start, STATUS_COLORS[status]], [stop, STATUS_COLORS[status]]])
    return scale


def _figure_div(figure, *, include_plotlyjs: bool = False) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotlyjs else False,
        config={"displaylogo": False, "responsive": True},
    )


def _review_table(records: list[TaskOutcomeRecord]) -> str:
    if not records:
        return '<p class="empty">No episodes require review.</p>'
    rows = []
    for record in records:
        keyframes = escape(record.keyframe_dir or "-")
        confidence = _confidence(record)
        rows.append(
            f'<tr data-episode="{escape(record.episode_id)}" '
            f'data-model-status="{escape(record.status)}" '
            f'data-confidence="{escape(confidence)}" '
            f'data-keyframes="{keyframes}">'
            f"<td>{escape(record.episode_id)}</td>"
            f'<td><span class="status {escape(record.status)}">'
            f"{escape(record.status.replace('_', ' '))}</span></td>"
            f"<td>{confidence}</td>"
            f"<td>{keyframes}</td>"
            "<td><select class='review-select' aria-label='Review status'>"
            "<option value='pending'>Pending</option>"
            "<option value='accept'>Accept</option><option value='reject'>Reject</option>"
            "<option value='uncertain'>Uncertain</option>"
            "</select></td></tr>"
        )
    return (
        "<table><thead><tr><th>Episode</th><th>Outcome</th><th>Confidence</th>"
        "<th>Keyframes</th><th>Review</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _document(
    *,
    title: str,
    subtitle: str,
    cards: tuple[tuple[str, str, str], ...],
    status_figure: str,
    confidence_figure: str,
    review_table: str,
) -> str:
    card_html = "".join(
        f"<div class='metric'><span>{escape(label)}</span><strong>{escape(value)}</strong>"
        f"<small>{escape(detail)}</small></div>"
        for label, value, detail in cards
    )
    storage_key = json.dumps(f"dem.task-reviews:{title}:{subtitle}")
    dataset = json.dumps(title.removesuffix(" Task Outcomes"))
    dashboard = json.dumps(subtitle)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
:root{{--ink:#26332f;--muted:#64716d;--paper:#f4f6f3;--panel:#fff;--line:#d8dfdb}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#eef3ef,#f7f3ea);color:var(--ink);font-family:"IBM Plex Sans","Aptos",sans-serif}}
main{{max-width:1320px;margin:auto;padding:32px 24px 48px}}h1{{margin:0;font-family:"IBM Plex Serif",Georgia,serif;font-size:30px;letter-spacing:0}}h2{{font-size:18px;margin:0 0 10px}}p{{color:var(--muted);overflow-wrap:anywhere}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}}.metric{{background:var(--panel);border:1px solid var(--line);padding:16px;border-radius:6px}}.metric span,.metric small{{display:block;color:var(--muted)}}.metric strong{{display:block;font-size:28px;margin:5px 0}}
section{{background:var(--panel);border-top:3px solid #287271;margin-top:16px;padding:18px}}.section-head{{display:flex;align-items:center;justify-content:space-between;gap:12px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{font-size:12px;text-transform:uppercase;color:var(--muted)}}
.status{{font-weight:700}}.inside{{color:#2f7d5b}}.uncertain{{color:#a66e1f}}.not_inside{{color:#c84b31}}.unparsed{{color:#7b8581}}select,button{{padding:7px 10px;border:1px solid var(--line);background:white;color:var(--ink)}}button{{cursor:pointer;font-weight:700}}button:disabled{{cursor:not-allowed;opacity:.45}}.empty{{margin:0}}#review-save-status{{font-size:13px;color:var(--muted);margin:8px 0}}
@media(max-width:720px){{main{{padding:20px 10px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><h1>{escape(title)}</h1><p>{escape(subtitle)}</p><div class="cards">{card_html}</div>
<section><h2>Outcome by episode</h2>{status_figure}</section>
<section><h2>Model confidence</h2>{confidence_figure}</section>
<section><div class="section-head"><h2>Review queue</h2><button id="export-reviews" type="button">Export review JSON</button></div><p>Uncertain, failed, unparsed, or confidence below 0.80.</p><p id="review-save-status">No saved reviews.</p><div class="table-wrap">{review_table}</div></section>
</main><script>
const reviewStorageKey = {storage_key};
const reviewDataset = {dataset};
const reviewDashboard = {dashboard};
const reviewSelects = Array.from(document.querySelectorAll('.review-select'));
let savedReviews = {{}};
try {{ savedReviews = JSON.parse(localStorage.getItem(reviewStorageKey) || '{{}}'); }} catch (_) {{ savedReviews = {{}}; }}

function updateReviewStatus() {{
    const count = Object.keys(savedReviews).length;
    document.getElementById('review-save-status').textContent = count
        ? `${{count}} review${{count === 1 ? '' : 's'}} saved in this browser.`
        : 'No saved reviews.';
    document.getElementById('export-reviews').disabled = count === 0;
}}

reviewSelects.forEach(function(select) {{
    const row = select.closest('tr');
    const episode = row.dataset.episode;
    if (savedReviews[episode]) select.value = savedReviews[episode].decision;
    select.addEventListener('change', function() {{
        if (select.value === 'pending') {{
            delete savedReviews[episode];
        }} else {{
            savedReviews[episode] = {{
                episode_id: episode,
                model_status: row.dataset.modelStatus,
                confidence: row.dataset.confidence === 'n/a' ? null : Number(row.dataset.confidence),
                keyframe_dir: row.dataset.keyframes === '-' ? null : row.dataset.keyframes,
                decision: select.value,
                reviewed_at: new Date().toISOString()
            }};
        }}
        localStorage.setItem(reviewStorageKey, JSON.stringify(savedReviews));
        updateReviewStatus();
    }});
}});

document.getElementById('export-reviews').addEventListener('click', function() {{
    const payload = {{
        dataset: reviewDataset,
        dashboard: reviewDashboard,
        exported_at: new Date().toISOString(),
        reviews: Object.values(savedReviews).sort((a, b) => a.episode_id.localeCompare(b.episode_id))
    }};
    const anchor = document.createElement('a');
    anchor.href = 'data:application/json;charset=utf-8,'
        + encodeURIComponent(JSON.stringify(payload, null, 2));
    anchor.download = `${{reviewDataset}}_task_reviews.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
}});

updateReviewStatus();
</script></body></html>"""