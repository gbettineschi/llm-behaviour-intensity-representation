"""Blind triplet-ranking annotation tool for validating the ordinal dataset.

An annotator types a username, then for each scenario orders three paraphrases
(negative / neutral / positive intensity) from lowest to highest trait intensity
— without seeing the labels. After each submission the correct order is revealed and the
answer is appended to data/<dir>/annotations/<username>.jsonl. On "Finish",
agreement between the annotator and the pipeline's accepted labels is computed
and shown (exact-order accuracy, pairwise accuracy, mean Kendall's tau).

Runs a local web app using only the Python standard library — no extra
dependencies, no install step.

Usage:
  python scripts/annotate.py [--data data/v2] [--data-root data] [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_LEVELS = ["negative", "neutral", "positive"]

# How to phrase the ordering axis for each trait in the UI.
TRAIT_PROMPT = {
    "politeness": "politeness — least polite to most polite",
    "hedging_confidence": "confidence — least confident (most hedged) to most confident",
}

ROOT = Path(__file__).resolve().parents[1]

# Module-level state, set in main().
DATA_ROOT: Path = ROOT / "data"
DEFAULT_DATA: str = "data/v2"
DATASET_CACHE: dict[str, dict] = {}


# ── Data ───────────────────────────────────────────────────────────────────

def to_display_path(path: Path) -> str:
  try:
    return path.relative_to(ROOT).as_posix()
  except ValueError:
    return path.as_posix()


def infer_levels(intensities: set[str]) -> list[str]:
  if {"low", "mid", "high"}.issubset(intensities):
    return ["low", "mid", "high"]
  if {"low", "medium", "high"}.issubset(intensities):
    return ["low", "medium", "high"]
  if {"negative", "neutral", "positive"}.issubset(intensities):
    return ["negative", "neutral", "positive"]
  raise ValueError(
    "unrecognized intensity labels in dataset: "
    + ", ".join(sorted(x for x in intensities if x is not None))
  )


def dataset_sources(data_dir: Path) -> tuple[str, list[Path]]:
  prompts_path = data_dir / "prompts.json"
  if prompts_path.exists():
    return "prompts_json", [prompts_path]
  accepted_paths = sorted(data_dir.glob("*/accepted.jsonl"))
  if accepted_paths:
    return "accepted_jsonl", accepted_paths
  return "none", []


def has_dataset_files(data_dir: Path) -> bool:
  kind, _paths = dataset_sources(data_dir)
  return kind != "none"


def load_prompt_records(data_dir: Path) -> list[dict]:
  kind, paths = dataset_sources(data_dir)
  if kind == "prompts_json":
    records = json.loads(paths[0].read_text())
    return [
      {
        "trait": r["trait"],
        "scenario_id": r["scenario_id"],
        "intensity": r["intensity"],
        "prompt": r["prompt"],
      }
      for r in records
    ]
  if kind == "accepted_jsonl":
    out: list[dict] = []
    for p in paths:
      for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
          continue
        r = json.loads(line)
        out.append(
          {
            "trait": r["trait"],
            "scenario_id": r["scenario_id"],
            "intensity": r["level"],
            "prompt": r["text"],
          }
        )
    return out
  raise FileNotFoundError(
    f"No dataset files found in {to_display_path(data_dir)} (expected prompts.json or */accepted.jsonl)."
  )


def build_triplets(data_dir: Path) -> tuple[list[dict], list[str], list[str]]:
  """One triplet per scenario that has a prompt at all three levels."""
  records = load_prompt_records(data_dir)
  levels = infer_levels({r.get("intensity") for r in records})

  grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
    lambda: defaultdict(list)
  )
  for r in records:
    grouped[(r["trait"], r["scenario_id"])][r["intensity"]].append(r["prompt"])

  triplets: list[dict] = []
  traits: set[str] = set()
  for (trait, sid), by_level in sorted(grouped.items()):
    if not all(by_level.get(lvl) for lvl in levels):
      continue  # need all three levels to form a triplet

    # Deterministic paraphrase pick + deterministic display shuffle, so the
    # task is stable across reloads but not trivially ordered.
    chosen = {lvl: sorted(by_level[lvl])[0] for lvl in levels}
    display = levels[:]
    random.Random(f"{trait}:{sid}").shuffle(display)
    cards = [
      {"cid": f"c{i}", "level": lvl, "text": chosen[lvl]}
      for i, lvl in enumerate(display)
    ]
    triplets.append(
      {
        "task_id": f"{trait}::{sid}",
        "trait": trait,
        "scenario_id": sid,
        "cards": cards,
      }
    )
    traits.add(trait)

  # Interleave traits so the annotator doesn't do one trait then the other.
  random.Random(20260521).shuffle(triplets)
  return triplets, sorted(traits), levels


def list_data_dirs(data_root: Path) -> list[str]:
  if not data_root.exists():
    return []
  out: list[str] = []
  for p in sorted(data_root.iterdir()):
    if not p.is_dir():
      continue
    if has_dataset_files(p):
      out.append(to_display_path(p))
  return out


def resolve_data_dir(data: str) -> Path:
  if not data:
    raise ValueError("missing data folder")
  path = Path(data)
  if not path.is_absolute():
    path = ROOT / path
  if not path.exists() or not has_dataset_files(path):
    raise ValueError(
      f"{to_display_path(path)} does not look like a dataset (expected prompts.json or */accepted.jsonl)."
    )
  return path


def load_dataset(data_dir: Path) -> dict:
  key = str(data_dir.resolve())
  if key in DATASET_CACHE:
    return DATASET_CACHE[key]

  triplets, traits, levels = build_triplets(data_dir)
  dataset = {
    "triplets": triplets,
    "triplet_by_id": {t["task_id"]: t for t in triplets},
    "available_traits": traits,
    "levels": levels,
    "annotations_dir": data_dir / "annotations",
  }
  DATASET_CACHE[key] = dataset
  return dataset


def slug(name: str) -> str:
  s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
  return s.strip("_")[:60] or "anon"


def annotation_path(username: str, data_dir: Path) -> Path:
  return data_dir / "annotations" / f"{slug(username)}.jsonl"


def read_annotations(username: str, data_dir: Path) -> list[dict]:
  path = annotation_path(username, data_dir)
  if not path.exists():
    return []
  return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# ── Agreement analysis ─────────────────────────────────────────────────────

def _stats(records: list[dict]) -> dict:
  n = len(records)
  if n == 0:
    return {"n": 0}

  exact = 0
  pair_correct = 0
  pair_total = 0
  taus: list[float] = []

  for r in records:
    gold_order = r.get("gold_order") or DEFAULT_LEVELS
    gold_rank = {lvl: i for i, lvl in enumerate(gold_order)}
    human_order = r["human_order"]
    human_rank = {lvl: i for i, lvl in enumerate(human_order)}

    if human_order == gold_order:
      exact += 1

    concordant = 0
    discordant = 0
    for a, b in combinations(gold_order, 2):
      gold_lt = gold_rank[a] < gold_rank[b]
      human_lt = human_rank[a] < human_rank[b]
      if gold_lt == human_lt:
        concordant += 1
        pair_correct += 1
      else:
        discordant += 1
      pair_total += 1

    denom = (len(gold_order) * (len(gold_order) - 1)) / 2
    taus.append((concordant - discordant) / (denom or 1.0))

  return {
    "n": n,
    "exact_order_accuracy": round(exact / n, 4),
    "pairwise_accuracy": round(pair_correct / pair_total, 4),
    "mean_kendall_tau": round(sum(taus) / len(taus), 4),
  }


def compute_agreement(records: list[dict]) -> dict:
    """Agreement between the annotator's orderings and the pipeline gold labels."""
    if not records:
        return {"n": 0}
    out = {"overall": _stats(records)}
    per_trait: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        per_trait[r["trait"]].append(r)
    for trait, recs in sorted(per_trait.items()):
        out[trait] = _stats(recs)
    return out


# ── HTTP server ────────────────────────────────────────────────────────────

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trait-intensity annotation</title>
<style>
  :root { font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  body { margin: 0; background: #f4f4f6; color: #1c1c1e; }
  .wrap { max-width: 720px; margin: 0 auto; padding: 24px 18px 60px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #6b6b70; font-size: 13px; margin-bottom: 20px; }
  .panel { background: #fff; border: 1px solid #e3e3e6; border-radius: 12px;
           padding: 20px; }
  input[type=text], select { font-size: 15px; padding: 9px 11px; width: 100%;
           box-sizing: border-box; border: 1px solid #c8c8cc; border-radius: 8px; }
  select { background: #fff; }
  .field { margin-top: 12px; }
  .field label { display: block; font-size: 12px; color: #6b6b70; margin: 0 0 6px; }
  .traits { display: flex; flex-direction: column; gap: 6px; }
  .trait-item { display: flex; gap: 8px; align-items: center; font-size: 14px; }
  .error { margin-top: 10px; font-size: 12px; color: #9c2a24; }
  button { font-size: 14px; padding: 9px 16px; border-radius: 8px;
           border: 1px solid #c8c8cc; background: #fff; cursor: pointer; }
  button.primary { background: #2b6cff; color: #fff; border-color: #2b6cff; }
  button:disabled { opacity: .45; cursor: default; }
  .progress { font-size: 13px; color: #6b6b70; margin-bottom: 12px; }
  .axis { font-size: 14px; margin: 6px 0 16px; }
  .axis b { color: #2b6cff; }
  .card { border: 2px solid #e3e3e6; border-radius: 10px; padding: 14px 14px;
          margin: 10px 0; cursor: pointer; display: flex; gap: 12px;
          align-items: flex-start; background: #fff; transition: border-color .1s; }
  .card:hover { border-color: #b9c8ff; }
  .card .badge { min-width: 26px; height: 26px; border-radius: 50%;
          background: #ececef; color: #6b6b70; font-weight: 600; font-size: 13px;
          display: flex; align-items: center; justify-content: center; }
  .card.ranked .badge { background: #2b6cff; color: #fff; }
  .card .text { flex: 1; line-height: 1.45; font-size: 15px; }
  .card .tag { font-size: 12px; color: #6b6b70; margin-top: 6px; }
  .card.good { border-color: #2faa55; }
  .card.bad  { border-color: #e0463e; }
  .rank-note { font-size: 12px; color: #6b6b70; margin-top: 4px; }
  .row { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
  .row .spacer { flex: 1; }
  .verdict { font-size: 14px; margin: 14px 0 0; padding: 10px 12px;
             border-radius: 8px; }
  .verdict.ok { background: #e5f6ea; color: #1c6b35; }
  .verdict.no { background: #fde8e7; color: #9c2a24; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 10px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #eee; }
  th { color: #6b6b70; font-weight: 600; }
  .big { font-size: 28px; font-weight: 700; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Trait-intensity annotation</h1>
  <div class="sub">Order three sentences from lowest to highest trait intensity.</div>

  <div id="login" class="panel">
    <p style="margin-top:0;font-size:14px">Enter your name or initials to begin. Progress is saved as you go.</p>
    <div class="field">
      <label for="username">Username</label>
      <input type="text" id="username" placeholder="username" autofocus>
    </div>
    <div class="field hidden" id="dataField">
      <label for="dataDir">Data folder</label>
      <select id="dataDir"></select>
    </div>
    <div class="field hidden" id="traitsField">
      <label>Traits</label>
      <div id="traits" class="traits"></div>
    </div>
    <div id="loginError" class="error hidden"></div>
    <div class="row"><button class="primary" id="startBtn" disabled>Start</button></div>
  </div>

  <div id="task" class="panel hidden">
    <div class="progress" id="progress"></div>
    <div class="axis">Order by <b id="axis"></b>.</div>
    <div id="cards"></div>
    <div class="rank-note">Click the sentences in order: first = lowest intensity, last = highest.</div>
    <div id="verdict" class="verdict hidden"></div>
    <div class="row">
      <button id="resetBtn">Reset</button>
      <button class="primary" id="submitBtn" disabled>Submit</button>
      <span class="spacer"></span>
      <button id="nextBtn" class="hidden">Next &rarr;</button>
      <button id="finishBtn">Finish &amp; see results</button>
    </div>
  </div>

  <div id="done" class="panel hidden">
    <h1 style="margin-top:0">Agreement with the dataset</h1>
    <div id="results"></div>
    <div class="row"><button id="backBtn">Back to annotating</button></div>
  </div>
</div>
<script>
let username = "";
let dataDir = "";
let tasks = [];
let idx = 0;
let ranking = [];      // cids in click order (negative -> positive)
let current = null;
let revealed = false;
let traitsLoadedFor = "";

const $ = (id) => document.getElementById(id);
const traitLabels = {
  politeness: "politeness",
  hedging_confidence: "confidence (hedging)",
};

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  let payload;
  try {
    payload = await r.json();
  } catch {
    payload = { error: "bad response" };
  }
  if (!r.ok && !payload.error) {
    payload.error = "request failed";
  }
  return payload;
}

function setLoginError(msg) {
  const box = $("loginError");
  if (!msg) {
    box.textContent = "";
    box.classList.add("hidden");
    return;
  }
  box.textContent = msg;
  box.classList.remove("hidden");
}

function updateStartButton() {
  const u = $("username").value.trim();
  const data = $("dataDir").value;
  const anyTrait =
    document.querySelectorAll('input[name="trait"]:checked').length > 0;
  $("startBtn").disabled = !(u && data && anyTrait);
}

function updateLoginVisibility() {
  const u = $("username").value.trim();
  if (!u) {
    $("dataField").classList.add("hidden");
    $("traitsField").classList.add("hidden");
    traitsLoadedFor = "";
    renderTraits([]);
    setLoginError("");
    updateStartButton();
    return;
  }
  $("dataField").classList.remove("hidden");
  const data = $("dataDir").value;
  if (!data) {
    $("traitsField").classList.add("hidden");
    renderTraits([]);
    updateStartButton();
    return;
  }
  $("traitsField").classList.remove("hidden");
  updateStartButton();
}

function renderDataDirs(dirs, defaultDir) {
  const sel = $("dataDir");
  sel.innerHTML = "";
  for (const d of dirs) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    sel.appendChild(opt);
  }
  if (defaultDir && dirs.includes(defaultDir)) {
    sel.value = defaultDir;
  } else if (dirs.length > 0) {
    sel.selectedIndex = 0;
  }
}

function renderTraits(traits) {
  const box = $("traits");
  box.innerHTML = "";
  if (!traits || traits.length === 0) {
    const msg = document.createElement("div");
    msg.className = "sub";
    msg.textContent = "No traits available for this data folder.";
    box.appendChild(msg);
    return;
  }
  for (const trait of traits) {
    const row = document.createElement("label");
    row.className = "trait-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "trait";
    input.value = trait;
    input.checked = true;
    input.onchange = updateStartButton;
    const label = document.createElement("span");
    label.textContent = traitLabels[trait] || trait;
    row.appendChild(input);
    row.appendChild(label);
    box.appendChild(row);
  }
}

async function loadTraits() {
  const data = $("dataDir").value;
  if (!data) {
    renderTraits([]);
    updateStartButton();
    return;
  }
  if (traitsLoadedFor === data) {
    updateStartButton();
    return;
  }
  const res = await api("/api/traits?data=" + encodeURIComponent(data));
  if (res.error) {
    setLoginError(res.error);
    renderTraits([]);
    updateStartButton();
    return;
  }
  setLoginError("");
  traitsLoadedFor = data;
  renderTraits(res.traits || []);
  updateStartButton();
}

async function initLogin() {
  const res = await api("/api/data-dirs");
  if (res.error) {
    setLoginError(res.error);
    return;
  }
  const dirs = res.data_dirs || [];
  renderDataDirs(dirs, res.default);
  if (dirs.length === 0) {
    setLoginError("No data folders with prompts.json found.");
    updateStartButton();
    return;
  }
  $("username").oninput = async () => {
    updateLoginVisibility();
    await loadTraits();
  };
  $("dataDir").onchange = async () => {
    traitsLoadedFor = "";
    updateLoginVisibility();
    await loadTraits();
  };
  updateLoginVisibility();
}

$("startBtn").onclick = async () => {
  const u = $("username").value.trim();
  const data = $("dataDir").value;
  const traits = Array.from(
    document.querySelectorAll('input[name="trait"]:checked')
  ).map((el) => el.value);
  if (!u || !data || traits.length === 0) {
    updateStartButton();
    if (!u) $("username").focus();
    return;
  }
  username = u;
  dataDir = data;
  setLoginError("");
  const dataRes = await api(
    "/api/tasks?username=" + encodeURIComponent(u) +
    "&data=" + encodeURIComponent(dataDir) +
    "&traits=" + encodeURIComponent(traits.join(","))
  );
  if (dataRes.error) {
    setLoginError(dataRes.error);
    return;
  }
  tasks = dataRes.tasks || [];
  $("login").classList.add("hidden");
  if (tasks.length === 0) { showDone(); return; }
  $("task").classList.remove("hidden");
  idx = 0;
  renderTask();
};

function renderTask() {
  revealed = false;
  ranking = [];
  current = tasks[idx];
  $("progress").textContent =
    "Triplet " + (idx + 1) + " of " + tasks.length;
  $("axis").textContent = current.axis;
  $("verdict").classList.add("hidden");
  $("nextBtn").classList.add("hidden");
  $("submitBtn").classList.remove("hidden");
  $("resetBtn").classList.remove("hidden");
  $("submitBtn").disabled = true;
  const box = $("cards");
  box.innerHTML = "";
  for (const card of current.cards) {
    const el = document.createElement("div");
    el.className = "card";
    el.dataset.cid = card.cid;
    el.innerHTML =
      '<div class="badge">&middot;</div>' +
      '<div class="text">' + escapeHtml(card.text) +
      '<div class="tag"></div></div>';
    el.onclick = () => clickCard(card.cid);
    box.appendChild(el);
  }
}

function clickCard(cid) {
  if (revealed) return;
  if (ranking.includes(cid)) return;
  ranking.push(cid);
  paintRanks();
  $("submitBtn").disabled = ranking.length !== 3;
}

function paintRanks() {
  const labels = { 0: "1 · lowest", 1: "2", 2: "3 · highest" };
  for (const el of document.querySelectorAll(".card")) {
    const pos = ranking.indexOf(el.dataset.cid);
    const badge = el.querySelector(".badge");
    if (pos === -1) {
      el.classList.remove("ranked");
      badge.textContent = "·";
    } else {
      el.classList.add("ranked");
      badge.textContent = labels[pos];
    }
  }
}

$("resetBtn").onclick = () => {
  if (revealed) return;
  ranking = [];
  paintRanks();
  $("submitBtn").disabled = true;
};

$("submitBtn").onclick = async () => {
  if (ranking.length !== 3) return;
  const res = await api("/api/answer",
    { username, data: dataDir, task_id: current.task_id, ranking });
  revealed = true;
  $("submitBtn").classList.add("hidden");
  $("resetBtn").classList.add("hidden");
  const goldOrder = res.gold_order || ["negative", "neutral", "positive"];
  const goldPosByLevel = {};
  for (let i = 0; i < goldOrder.length; i += 1) goldPosByLevel[goldOrder[i]] = i;
  // res.levels: { cid: level }
  for (const el of document.querySelectorAll(".card")) {
    const cid = el.dataset.cid;
    const level = res.levels[cid];
    const humanPos = ranking.indexOf(cid);
    const goldPos = goldPosByLevel[level];
    el.classList.add(humanPos === goldPos ? "good" : "bad");
    el.querySelector(".tag").textContent =
      "true level: " + level.toUpperCase();
  }
  const v = $("verdict");
  v.classList.remove("hidden");
  if (res.exact) {
    v.className = "verdict ok";
    v.textContent = "Correct — your order matches the dataset.";
  } else {
    v.className = "verdict no";
    v.textContent = "Not quite — " + res.pairwise_correct +
      "/3 pairs ordered correctly. Correct order shown above.";
  }
  $("nextBtn").classList.remove("hidden");
  if (idx + 1 >= tasks.length) $("nextBtn").textContent = "See results";
};

$("nextBtn").onclick = () => {
  idx += 1;
  if (idx >= tasks.length) { showDone(); return; }
  renderTask();
};

$("finishBtn").onclick = showDone;
$("backBtn").onclick = () => {
  $("done").classList.add("hidden");
  if (idx < tasks.length) $("task").classList.remove("hidden");
};

async function showDone() {
  $("task").classList.add("hidden");
  $("login").classList.add("hidden");
  const box = $("results");
  const data = await api("/api/finish", { username, data: dataDir });
  if (data.error) {
    box.innerHTML = "<p>" + escapeHtml(data.error) + "</p>";
    $("done").classList.remove("hidden");
    return;
  }
  const r = data.agreement;
  if (!r || !r.n && !r.overall) {
    box.innerHTML = "<p>No answers recorded yet.</p>";
  } else {
    const o = r.overall;
    let html =
      '<p class="big">' + pct(o.exact_order_accuracy) + '</p>' +
      '<p class="sub">exact-order agreement over ' + o.n + ' triplets</p>' +
      '<table><tr><th>scope</th><th>n</th><th>exact order</th>' +
      '<th>pairwise</th><th>Kendall &tau;</th></tr>';
    html += rowFor("overall", r.overall);
    for (const k of Object.keys(r)) {
      if (k === "overall") continue;
      html += rowFor(k, r[k]);
    }
    html += '</table>';
    if (data.saved_to) {
      html += '<p class="sub" style="margin-top:14px">Saved to ' +
        escapeHtml(data.saved_to) + '</p>';
    }
    box.innerHTML = html;
  }
  $("done").classList.remove("hidden");
}

function rowFor(name, s) {
  return '<tr><td>' + escapeHtml(name) + '</td><td>' + s.n + '</td><td>' +
    pct(s.exact_order_accuracy) + '</td><td>' + pct(s.pairwise_accuracy) +
    '</td><td>' + s.mean_kendall_tau.toFixed(3) + '</td></tr>';
}
function pct(x) { return (100 * x).toFixed(1) + "%"; }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

window.addEventListener("load", initLogin);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/data-dirs":
            data_dirs = list_data_dirs(DATA_ROOT)
            default_data = DEFAULT_DATA
            if default_data and default_data not in data_dirs:
                try:
                    default_path = resolve_data_dir(default_data)
                except ValueError:
                    default_path = None
                if default_path:
                    default_data = to_display_path(default_path)
                    data_dirs.append(default_data)
            self._json({"data_dirs": data_dirs, "default": default_data})
        elif parsed.path == "/api/traits":
            data = parse_qs(parsed.query).get("data", [""])[0]
            try:
                data_dir = resolve_data_dir(data)
                dataset = load_dataset(data_dir)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json({"traits": dataset["available_traits"]})
        elif parsed.path == "/api/tasks":
            query = parse_qs(parsed.query)
            username = query.get("username", [""])[0]
            data = query.get("data", [""])[0]
            traits_raw = query.get("traits", [""])[0]
            try:
                data_dir = resolve_data_dir(data)
                dataset = load_dataset(data_dir)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
            if not dataset["triplets"]:
                self._json(
                    {
                        "error": (
                            f"No complete triplets in {to_display_path(data_dir)}/prompts.json "
                            "(need a scenario with accepted prompts at all three levels)."
                        )
                    },
                    400,
                )
                return
            trait_set = {t for t in traits_raw.split(",") if t}
            selected = dataset["triplets"]
            if trait_set:
                selected = [t for t in selected if t["trait"] in trait_set]
            done = {r["task_id"] for r in read_annotations(username, data_dir)}
            tasks = []
            for t in selected:
                if t["task_id"] in done:
                    continue
                tasks.append(
                    {
                        "task_id": t["task_id"],
                        "axis": TRAIT_PROMPT.get(t["trait"], t["trait"]),
                        # Blind: cid + text only, no level.
                        "cards": [{"cid": c["cid"], "text": c["text"]} for c in t["cards"]],
                    }
                )
            self._json({"tasks": tasks})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        if self.path == "/api/answer":
            self._handle_answer(body)
        elif self.path == "/api/finish":
            self._handle_finish(body)
        else:
            self._send(404, b"not found", "text/plain")

    def _handle_answer(self, body: dict) -> None:
        username = body.get("username", "")
        data = body.get("data", "")
        task_id = body.get("task_id", "")
        ranking = body.get("ranking", [])
        try:
            data_dir = resolve_data_dir(data)
            dataset = load_dataset(data_dir)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
            return
        triplet = dataset["triplet_by_id"].get(task_id)
        if not username or triplet is None or len(ranking) != 3:
            self._json({"error": "invalid answer"}, 400)
            return
        cid_to_level = {c["cid"]: c["level"] for c in triplet["cards"]}
        if set(ranking) != set(cid_to_level):
            self._json({"error": "ranking does not match cards"}, 400)
            return
        gold_order = dataset.get("levels") or DEFAULT_LEVELS
        gold_rank = {lvl: i for i, lvl in enumerate(gold_order)}
        # human_order: levels ordered low->high (or negative->positive) as the annotator placed them.
        human_order = [cid_to_level[cid] for cid in ranking]
        exact = human_order == gold_order
        human_rank = {lvl: i for i, lvl in enumerate(human_order)}
        pair_correct = sum(
            1
          for a, b in combinations(gold_order, 2)
          if (gold_rank[a] < gold_rank[b]) == (human_rank[a] < human_rank[b])
        )
        record = {
            "username": username,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_dir": to_display_path(data_dir),
            "task_id": task_id,
            "trait": triplet["trait"],
            "scenario_id": triplet["scenario_id"],
            "displayed_cids": [c["cid"] for c in triplet["cards"]],
            "human_order": human_order,
            "gold_order": gold_order,
            "exact_correct": exact,
            "pairwise_correct": pair_correct,
        }
        path = annotation_path(username, data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._json(
            {
                "levels": cid_to_level,
            "gold_order": gold_order,
                "exact": exact,
                "pairwise_correct": pair_correct,
            }
        )

    def _handle_finish(self, body: dict) -> None:
      username = body.get("username", "")
      data = body.get("data", "")
      try:
        data_dir = resolve_data_dir(data)
      except ValueError as exc:
        self._json({"error": str(exc)}, 400)
        return

      records = read_annotations(username, data_dir)
      agreement = compute_agreement(records)
      summary = {
        "username": username,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": to_display_path(data_dir),
        "agreement": agreement,
      }

      saved_to = ""
      if records:
        summary_path = data_dir / "annotations" / f"{slug(username)}__summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        saved_to = str(summary_path)
      self._json({"agreement": agreement, "saved_to": saved_to})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/v2",
        help="Generation output dir containing prompts.json (default: data/v2)",
    )
    parser.add_argument(
      "--data-root",
      default="data",
      help="Root folder containing dataset subdirectories (default: data)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open a browser")
    args = parser.parse_args()

    global DATA_ROOT, DEFAULT_DATA
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
      data_root = ROOT / data_root
    DATA_ROOT = data_root

    default_path = Path(args.data)
    if not default_path.is_absolute():
      default_path = ROOT / default_path
    default_exists = default_path.exists() and (default_path / "prompts.json").exists()
    DEFAULT_DATA = to_display_path(default_path) if default_exists else ""

    data_dirs = list_data_dirs(DATA_ROOT)
    if DEFAULT_DATA and DEFAULT_DATA not in data_dirs:
      data_dirs.append(DEFAULT_DATA)

    url = f"http://localhost:{args.port}/"
    if not default_exists:
      print(f"Warning: default dataset {to_display_path(default_path)} not found.")
    if data_dirs:
      print(f"Found {len(data_dirs)} dataset(s) under {to_display_path(DATA_ROOT)}.")
    else:
      print(f"No datasets found under {to_display_path(DATA_ROOT)}.")
    if DEFAULT_DATA:
      print(f"Default dataset: {DEFAULT_DATA}")
    print(f"Serving at {url}  (Ctrl-C to stop)")
    print("Select data folder and traits in the browser.")
    if not args.no_browser:
        webbrowser.open(url)
    server = ThreadingHTTPServer(("localhost", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
