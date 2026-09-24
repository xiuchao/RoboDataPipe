from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_io import DATASET_REGISTRY
from robehavior.online_monitor import (
  OnlineFailureMonitor,
  build_online_monitor_config,
  get_dataset_camera_candidates,
  iter_jsonl_samples,
  select_available_cameras,
)
from vlm.qwen_vl_config import PROMPT_MODES
from workflows.qwen_status import QwenObjectStatusJudge


def resolve_stream_paths(input_jsonl: str | None, input_dir: str | None) -> list[Path]:
  stream_paths: list[Path] = []
  if input_jsonl is not None:
    stream_paths.append(Path(input_jsonl))
  if input_dir is not None:
    stream_paths.extend(sorted(Path(input_dir).glob("ep_*.jsonl")))
  unique_paths: list[Path] = []
  seen: set[Path] = set()
  for path in stream_paths:
    resolved = path.resolve()
    if resolved in seen:
      continue
    seen.add(resolved)
    unique_paths.append(path)
  if not unique_paths:
    raise ValueError("No stream JSONL files found. Provide --input-jsonl or --input-dir with ep_*.jsonl files.")
  return unique_paths


HTML_PAGE = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Online Monitor GUI</title>
  <style>
    :root {
      --bg: #f3f0e8;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d6d0c4;
      --accent: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #fff7ed, var(--bg));
    }
    .shell {
      display: grid;
      grid-template-columns: minmax(360px, 1.3fr) minmax(320px, 0.9fr);
      gap: 18px;
      min-height: 100vh;
      padding: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 16px 50px rgba(31, 41, 55, 0.08);
    }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }
    h1, h2, h3 { margin: 0; font-weight: 600; }
    h1 { font-size: 28px; }
    .muted { color: var(--muted); }
    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin: 12px 0 16px;
    }
    button, select {
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      font: inherit;
    }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    button.warn {
      background: #fff7ed;
      border-color: #fdba74;
      color: #9a3412;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: #fffcf6;
    }
    .stat .label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .stat .value {
      font-size: 22px;
      margin-top: 6px;
    }
    .images {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }
    .image-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: #fff;
    }
    .image-card img {
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      border-radius: 10px;
      background: #ece7dc;
      transition: opacity 140ms ease;
    }
    .image-card img.is-loading {
      opacity: 0.88;
    }
    .image-card .caption {
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
      word-break: break-all;
    }
    .timeline, .alerts {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 72vh;
      overflow: auto;
      padding-right: 4px;
    }
    .event-item, .alert-item {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: white;
    }
    .event-item .head, .alert-item .head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
      font-size: 14px;
    }
    .event-tag, .alert-tag {
      font-weight: 700;
      color: var(--accent);
    }
    .alert-tag { color: var(--bad); }
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      margin-top: 8px;
      background: #faf5ef;
    }
    .status-inside { color: #166534; }
    .status-uncertain { color: var(--warn); }
    .status-bad { color: var(--bad); }
    .small { font-size: 13px; }
    @media (max-width: 1100px) {
      .shell { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <div class="hero">
        <div>
          <h1>Online Monitor GUI</h1>
          <div class="muted small" id="streamLabel"></div>
        </div>
        <div class="muted small" id="playState">paused</div>
      </div>

      <div class="controls">
        <button class="primary" id="playPauseButton" onclick="togglePlaying()">Play</button>
        <button onclick="stepFrame()">Step</button>
        <button class="warn" onclick="resetSession()">Reset</button>
        <span class="muted small">Speed</span>
        <select id="speedSelect" onchange="setPlaybackSpeed(this.value)">
          <option value="1000">1.0s</option>
          <option value="700" selected>0.7s</option>
          <option value="400">0.4s</option>
          <option value="250">0.25s</option>
          <option value="150">0.15s</option>
        </select>
      </div>

      <div class="stats">
        <div class="stat"><div class="label">Frame</div><div class="value" id="frameValue">-</div></div>
        <div class="stat"><div class="label">Timestamp</div><div class="value" id="timestampValue">-</div></div>
        <div class="stat"><div class="label">Events</div><div class="value" id="eventCount">0</div></div>
        <div class="stat"><div class="label">Alerts</div><div class="value" id="alertCount">0</div></div>
      </div>

      <div class="images" id="imageGrid"></div>
    </section>

    <section class="panel">
      <div style="display:grid; grid-template-columns: 1fr; gap: 18px;">
        <div>
          <h2>Event Timeline</h2>
          <div class="muted small" style="margin: 6px 0 12px;">Live event descriptions emitted from the online monitor.</div>
          <div class="timeline" id="timeline"></div>
        </div>
        <div>
          <h2>Alerts</h2>
          <div class="muted small" style="margin: 6px 0 12px;">Only uncertain and failure states appear here.</div>
          <div class="alerts" id="alerts"></div>
        </div>
      </div>
    </section>
  </div>

  <script>
    let playing = false;
    let timer = null;
    let playbackIntervalMs = 700;
    const imageCards = new Map();
    let lastRenderedEventCount = -1;
    let lastRenderedAlertCount = -1;

    function statusClass(status) {
      if (!status) return '';
      if (status === 'inside') return 'status-inside';
      if (status === 'uncertain') return 'status-uncertain';
      return 'status-bad';
    }

    function eventDescription(event) {
      const frame = `frame ${event.frame_index}`;
      if (event.event === 'initial_state') return event.message || `Initial state recorded at ${frame}.`;
      if (event.event === 'left_arm_still_confirmed') return event.message || `Left arm stationary start state confirmed at ${frame}.`;
      if (event.event === 'right_arm_start_moving') return `Right arm started moving at ${frame}.`;
      if (event.event === 'right_arm_stop_moving') return `Right arm stopped moving at ${frame}.`;
      if (event.event === 'right_arm_start_to_place') return `Right arm started moving at ${frame}.`;
      if (event.event === 'gripper_start_opening') return `Gripper started opening from action command at ${frame}.`;
      if (event.event === 'gripper_fully_open') return `Gripper reached fully-open state at ${frame}.`;
      if (event.event === 'release_retreat_start') return `Retreat after release started at ${frame}.`;
      if (event.event === 'object_in_shelf_status') return `Object shelf status judged as ${event.placement_status || 'unknown'} at ${frame}.`;
      return `${event.event} at ${frame}.`;
    }

    function compareByTime(left, right) {
      const leftFrame = Number.isFinite(left.frame_index) ? left.frame_index : Number.MAX_SAFE_INTEGER;
      const rightFrame = Number.isFinite(right.frame_index) ? right.frame_index : Number.MAX_SAFE_INTEGER;
      if (leftFrame !== rightFrame) {
        return leftFrame - rightFrame;
      }

      const leftTimestamp = Number.isFinite(left.timestamp) ? left.timestamp : Number.MAX_VALUE;
      const rightTimestamp = Number.isFinite(right.timestamp) ? right.timestamp : Number.MAX_VALUE;
      if (leftTimestamp !== rightTimestamp) {
        return leftTimestamp - rightTimestamp;
      }

      return String(left.event || '').localeCompare(String(right.event || ''));
    }

    function setPlaybackSpeed(value) {
      playbackIntervalMs = Number(value);
      if (playing) {
        setPlaying(true);
      }
    }

    function updatePlaybackControls() {
      document.getElementById('playState').textContent = playing ? 'playing' : 'paused';
      document.getElementById('playPauseButton').textContent = playing ? 'Pause' : 'Play';
    }

    function togglePlaying() {
      setPlaying(!playing);
    }

    async function api(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      return response.json();
    }

    async function stepFrame() {
      const state = await api('/api/step', { method: 'POST' });
      render(state);
      if (state.done) {
        setPlaying(false);
      }
    }

    async function resetSession() {
      setPlaying(false);
      const state = await api('/api/reset', { method: 'POST' });
      render(state);
    }

    function setPlaying(value) {
      playing = value;
      updatePlaybackControls();
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (playing) {
        scheduleNextFrame();
      }
    }

    function scheduleNextFrame() {
      if (!playing) {
        return;
      }
      timer = setTimeout(async () => {
        try {
          await stepFrame();
        } catch (error) {
          console.error(error);
          setPlaying(false);
          return;
        }
        scheduleNextFrame();
      }, playbackIntervalMs);
    }

    function buildImageCard(image) {
      const card = document.createElement('div');
      card.className = 'image-card';
      card.dataset.key = image.camera;

      const img = document.createElement('img');
      img.alt = image.camera;
      img.decoding = 'async';

      const caption = document.createElement('div');
      caption.className = 'caption';

      card.appendChild(img);
      card.appendChild(caption);
      return card;
    }

    function updateImageElement(img, nextUrl) {
      if (img.dataset.loadedSrc === nextUrl || img.dataset.pendingSrc === nextUrl) {
        return;
      }

      img.dataset.pendingSrc = nextUrl;
      img.classList.add('is-loading');

      const loader = new Image();
      loader.decoding = 'async';
      loader.onload = () => {
        if (img.dataset.pendingSrc !== nextUrl) {
          return;
        }
        img.src = nextUrl;
        img.dataset.loadedSrc = nextUrl;
        img.classList.remove('is-loading');
      };
      loader.onerror = () => {
        if (img.dataset.pendingSrc === nextUrl) {
          img.classList.remove('is-loading');
        }
      };
      loader.src = nextUrl;
    }

    function renderImages(images) {
      const grid = document.getElementById('imageGrid');
      const activeKeys = new Set();

      for (const image of images) {
        activeKeys.add(image.camera);
        let card = imageCards.get(image.camera);
        if (!card) {
          card = buildImageCard(image);
          imageCards.set(image.camera, card);
          grid.appendChild(card);
        }

        const img = card.querySelector('img');
        const caption = card.querySelector('.caption');
        img.alt = image.camera;
        updateImageElement(img, image.url);
        caption.textContent = `${image.camera} · ${image.path}`;
      }

      for (const [key, card] of imageCards.entries()) {
        if (activeKeys.has(key)) {
          if (!card.isConnected) {
            grid.appendChild(card);
          }
          continue;
        }
        imageCards.delete(key);
        card.remove();
      }
    }

    function renderEvents(events) {
      const timeline = document.getElementById('timeline');
      timeline.innerHTML = '';
      const sortedEvents = [...events].sort(compareByTime);
      for (const event of sortedEvents) {
        const item = document.createElement('div');
        item.className = 'event-item';
        const head = document.createElement('div');
        head.className = 'head';
        head.innerHTML = `<span class="event-tag ${statusClass(event.placement_status)}">${event.event}</span><span class="muted">frame ${event.frame_index}</span>`;
        const body = document.createElement('div');
        body.className = 'small';
        body.textContent = eventDescription(event);
        item.appendChild(head);
        item.appendChild(body);
        if (event.placement_status) {
          const badge = document.createElement('div');
          badge.className = `badge ${statusClass(event.placement_status)}`;
          badge.textContent = `placement_status=${event.placement_status} confidence=${event.confidence}`;
          item.appendChild(badge);
        }
        timeline.appendChild(item);
      }
    }

    function renderAlerts(alerts) {
      const alertsNode = document.getElementById('alerts');
      alertsNode.innerHTML = '';
      const sortedAlerts = [...alerts].sort(compareByTime);
      for (const alert of sortedAlerts) {
        const item = document.createElement('div');
        item.className = 'alert-item';
        item.innerHTML = `<div class="head"><span class="alert-tag">${alert.alert_type}</span><span class="muted">frame ${alert.frame_index}</span></div><div class="small">${alert.message}</div>`;
        alertsNode.appendChild(item);
      }
    }

    function render(state) {
      document.getElementById('streamLabel').textContent = `${state.dataset} · episode ${state.stream_index}/${state.total_streams} · ${state.stream_name} · sample ${state.current_index}/${state.total_samples}`;
      document.getElementById('frameValue').textContent = state.current_sample ? state.current_sample.frame_index : '-';
      document.getElementById('timestampValue').textContent = state.current_sample ? state.current_sample.timestamp : '-';
      document.getElementById('eventCount').textContent = state.events.length;
      document.getElementById('alertCount').textContent = state.alerts.length;
      renderImages(state.current_images || []);
      if (state.events.length !== lastRenderedEventCount) {
        renderEvents(state.events || []);
        lastRenderedEventCount = state.events.length;
      }
      if (state.alerts.length !== lastRenderedAlertCount) {
        renderAlerts(state.alerts || []);
        lastRenderedAlertCount = state.alerts.length;
      }
    }

    async function init() {
      const state = await api('/api/state');
      render(state);
    }

    init().catch((error) => console.error(error));
  </script>
</body>
</html>
'''


class MonitorSession:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self._lock = threading.RLock()
    self.stream_paths = resolve_stream_paths(args.input_jsonl, args.input_dir)
    self.stream_index = 0
    self.samples: list[dict[str, Any]] = []
    self.total_samples = 0
    self.event_log: list[dict[str, Any]] = []
    self.alerts: list[dict[str, Any]] = []
    self.current_index = -1
    self.current_sample: dict[str, Any] | None = None
    self.monitor: OnlineFailureMonitor | None = None
    self._load_stream(0)

  def _load_stream(self, stream_index: int) -> None:
    self.stream_index = stream_index
    self.samples = list(iter_jsonl_samples(self.stream_paths[stream_index]))
    requested_cameras = list(self.args.cameras or get_dataset_camera_candidates(self.args.dataset, self.args.registry))
    self.args.cameras = select_available_cameras(requested_cameras, self.samples)
    self.total_samples = len(self.samples)
    self.event_log = []
    self.alerts = []
    self.current_index = -1
    self.current_sample = None
    self.monitor = None

  def _build_monitor(self) -> OnlineFailureMonitor:
    config = build_online_monitor_config(self.args)
    return OnlineFailureMonitor(config, QwenObjectStatusJudge(config))

  def reset(self) -> dict[str, Any]:
    with self._lock:
      self._load_stream(0)
      return self.state()

  def step(self) -> dict[str, Any]:
    with self._lock:
      if self.current_index + 1 >= self.total_samples:
        if self.stream_index + 1 >= len(self.stream_paths):
          return self.state(done=True)
        self._load_stream(self.stream_index + 1)

      if self.monitor is None:
        self.monitor = self._build_monitor()

      self.current_index += 1
      self.current_sample = self.samples[self.current_index]
      update = self.monitor.add_sample(self.current_sample)
      if update is not None:
        self.event_log.extend(update.get("events", []))
        self.alerts.extend(update.get("alerts", []))
      return self.state(done=self.current_index + 1 >= self.total_samples)

  def state(self, *, done: bool = False) -> dict[str, Any]:
    with self._lock:
      return {
        "dataset": self.args.dataset,
        "stream_path": str(self.stream_paths[self.stream_index]),
        "stream_name": self.stream_paths[self.stream_index].name,
        "stream_index": self.stream_index + 1,
        "total_streams": len(self.stream_paths),
        "done": done,
        "current_index": self.current_index + 1 if self.current_index >= 0 else 0,
        "total_samples": self.total_samples,
        "current_sample": self.current_sample,
        "current_images": current_image_payload(self.current_sample),
        "events": self.event_log,
        "alerts": self.alerts,
      }


def current_image_payload(sample: dict[str, Any] | None) -> list[dict[str, str]]:
    if sample is None:
        return []
    payload: list[dict[str, str]] = []
    images = sample.get("images", {})
    if isinstance(images, dict):
        for camera, path in images.items():
            payload.append({
                "camera": camera,
                "path": str(path),
                "url": image_url_for_path(Path(path)),
            })
        return payload

    for key, value in sample.items():
        if key.startswith("observation.images."):
            payload.append({
                "camera": key,
                "path": str(value),
                "url": image_url_for_path(Path(str(value))),
            })
    return payload


def image_url_for_path(path: Path) -> str:
    return "/image?path=" + str(path).replace(" ", "%20")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", required=True, help="Dataset nickname in datasets.yaml, e.g. DEM_pickplace.")
  parser.add_argument("--input-jsonl", default=None, help="Path to a JSONL stream file to replay in the GUI.")
  parser.add_argument("--input-dir", default=None, help="Directory containing ep_*.jsonl stream files to replay sequentially in the GUI.")
  parser.add_argument("--registry", default=DATASET_REGISTRY, help="Path to datasets.yaml.")
  parser.add_argument(
    "--prompt-mode",
    choices=sorted(PROMPT_MODES.keys()),
    default="shelf_placement_after_release",
    help="Prompt template to use for online monitoring.",
  )
  parser.add_argument("--question", default=None, help="Optional explicit question override.")
  parser.add_argument("--camera", action="append", dest="cameras", default=None, help="Camera key to include. Can be repeated.")
  parser.add_argument("--model", default=None, help="Optional model override.")
  parser.add_argument("--max-new-tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
  parser.add_argument("--signal-source", default=None)
  parser.add_argument("--release-gripper-source", default=None)
  parser.add_argument("--release-open-tolerance", type=float, default=None)
  parser.add_argument("--release-stationary-window", type=int, default=None)
  parser.add_argument("--release-stationary-displacement-threshold", type=float, default=None)
  parser.add_argument("--approach-source", default=None)
  parser.add_argument("--approach-window", type=int, default=None)
  parser.add_argument("--approach-displacement-threshold", type=float, default=None)
  parser.add_argument("--max-buffer-frames", type=int, default=64)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()
  if bool(args.input_jsonl) == bool(args.input_dir):
    parser.error("Provide exactly one of --input-jsonl or --input-dir.")
  return args


class GUIHandler(BaseHTTPRequestHandler):
    session: MonitorSession

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/state":
            self._send_json(self.session.state())
            return
        if parsed.path == "/image":
            params = parse_qs(parsed.query)
            raw_path = params.get("path", [None])[0]
            if raw_path is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing image path")
                return
            self._send_image(Path(raw_path))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
      parsed = urlparse(self.path)
      if parsed.path == "/api/step":
        self._send_json(self.session.step())
        return
      if parsed.path == "/api/reset":
        self._send_json(self.session.reset())
        return
      self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, obj: Any) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_image(self, path: Path) -> None:
        resolved = path if path.is_absolute() else (PROJECT_ROOT / path)
        if not resolved.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Image not found: {resolved}")
            return
        suffix = resolved.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(suffix, "application/octet-stream")
        payload = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    args = parse_args()
    session = MonitorSession(args)
    GUIHandler.session = session
    server = ThreadingHTTPServer((args.host, args.port), GUIHandler)
    print(f"[gui] http://{args.host}:{args.port}", flush=True)
    if args.input_dir is not None:
        print(f"[stream_dir] {args.input_dir}", flush=True)
    else:
        print(f"[stream] {args.input_jsonl}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
