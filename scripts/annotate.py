"""Blind triplet-ranking annotation tool for validating the ordinal dataset.

An annotator types a username, then for each scenario orders three paraphrases
(low / mid / high intensity) from lowest to highest trait intensity — without
seeing the labels. After each submission the correct order is revealed and the
answer is appended to data/<dir>/annotations/<username>.jsonl. On "Finish",
agreement between the annotator and the pipeline's accepted labels is computed
and shown (exact-order accuracy, pairwise accuracy, mean Kendall's tau).

Runs a local web app using only the Python standard library — no extra
dependencies, no install step.

Usage:
    python scripts/annotate.py [--data data/v1_smoke_large] [--port 8765]
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

LEVELS = ["low", "mid", "high"]
GOLD_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

# How to phrase the ordering axis for each trait in the UI.
TRAIT_PROMPT = {
    "politeness": "politeness — least polite to most polite",
    "hedging_confidence": "confidence — least confident (most hedged) to most confident",
}

# Module-level state, set in main().
TRIPLETS: list[dict] = []
TRIPLET_BY_ID: dict[str, dict] = {}
ANNOTATIONS_DIR: Path = Path()


# ── Data ───────────────────────────────────────────────────────────────────

def build_triplets(data_dir: Path) -> list[dict]:
    """One triplet per scenario that has an accepted prompt at all three levels."""
    prompts_path = data_dir / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"{prompts_path} not found — run the generation pipeline first."
        )
    records = json.loads(prompts_path.read_text())
    grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        grouped[(r["trait"], r["scenario_id"])][r["intensity"]].append(r["prompt"])

    triplets: list[dict] = []
    for (trait, sid), by_level in sorted(grouped.items()):
        if not all(by_level.get(lvl) for lvl in LEVELS):
            continue  # need all three levels to form a triplet
        # Deterministic paraphrase pick + deterministic display shuffle, so the
        # task is stable across reloads but not trivially low-mid-high.
        chosen = {lvl: sorted(by_level[lvl])[0] for lvl in LEVELS}
        display = LEVELS[:]
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
    # Interleave traits so the annotator doesn't do one trait then the other.
    random.Random(20260521).shuffle(triplets)
    return triplets


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return s.strip("_")[:60] or "anon"


def annotation_path(username: str) -> Path:
    return ANNOTATIONS_DIR / f"{slug(username)}.jsonl"


def read_annotations(username: str) -> list[dict]:
    path = annotation_path(username)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


# ── Agreement analysis ─────────────────────────────────────────────────────

def _stats(records: list[dict]) -> dict:
    n = len(records)
    exact = 0
    pair_correct = pair_total = 0
    taus: list[float] = []
    for r in records:
        human_rank = {lvl: i for i, lvl in enumerate(r["human_order"])}
        if r["human_order"] == LEVELS:
            exact += 1
        concordant = discordant = 0
        for a, b in combinations(LEVELS, 2):
            gold_lt = GOLD_RANK[a] < GOLD_RANK[b]
            human_lt = human_rank[a] < human_rank[b]
            if gold_lt == human_lt:
                concordant += 1
                pair_correct += 1
            else:
                discordant += 1
            pair_total += 1
        taus.append((concordant - discordant) / 3.0)
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
  input[type=text] { font-size: 15px; padding: 9px 11px; width: 100%;
           box-sizing: border-box; border: 1px solid #c8c8cc; border-radius: 8px; }
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
    <input type="text" id="username" placeholder="username" autofocus>
    <div class="row"><button class="primary" id="startBtn">Start</button></div>
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
let tasks = [];
let idx = 0;
let ranking = [];      // cids in click order (low -> high)
let current = null;
let revealed = false;

const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  return r.json();
}

$("startBtn").onclick = async () => {
  const u = $("username").value.trim();
  if (!u) { $("username").focus(); return; }
  username = u;
  const data = await api("/api/tasks?username=" + encodeURIComponent(u));
  tasks = data.tasks;
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
    { username, task_id: current.task_id, ranking });
  revealed = true;
  $("submitBtn").classList.add("hidden");
  $("resetBtn").classList.add("hidden");
  // res.levels: { cid: level }
  for (const el of document.querySelectorAll(".card")) {
    const cid = el.dataset.cid;
    const level = res.levels[cid];
    const humanPos = ranking.indexOf(cid);
    const goldPos = { low: 0, mid: 1, high: 2 }[level];
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
  const data = await api("/api/finish", { username });
  const r = data.agreement;
  const box = $("results");
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
    html += '<p class="sub" style="margin-top:14px">Saved to ' +
      escapeHtml(data.saved_to) + '</p>';
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
        elif parsed.path == "/api/tasks":
            username = parse_qs(parsed.query).get("username", [""])[0]
            done = {r["task_id"] for r in read_annotations(username)}
            tasks = []
            for t in TRIPLETS:
                if t["task_id"] in done:
                    continue
                tasks.append(
                    {
                        "task_id": t["task_id"],
                        "axis": TRAIT_PROMPT.get(t["trait"], t["trait"]),
                        # Blind: cid + text only, no level.
                        "cards": [
                            {"cid": c["cid"], "text": c["text"]}
                            for c in t["cards"]
                        ],
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
        task_id = body.get("task_id", "")
        ranking = body.get("ranking", [])
        triplet = TRIPLET_BY_ID.get(task_id)
        if not username or triplet is None or len(ranking) != 3:
            self._json({"error": "invalid answer"}, 400)
            return
        cid_to_level = {c["cid"]: c["level"] for c in triplet["cards"]}
        if set(ranking) != set(cid_to_level):
            self._json({"error": "ranking does not match cards"}, 400)
            return
        # human_order: levels ordered low->high as the annotator placed them.
        human_order = [cid_to_level[cid] for cid in ranking]
        exact = human_order == LEVELS
        human_rank = {lvl: i for i, lvl in enumerate(human_order)}
        pair_correct = sum(
            1
            for a, b in combinations(LEVELS, 2)
            if (GOLD_RANK[a] < GOLD_RANK[b]) == (human_rank[a] < human_rank[b])
        )
        record = {
            "username": username,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "trait": triplet["trait"],
            "scenario_id": triplet["scenario_id"],
            "displayed_cids": [c["cid"] for c in triplet["cards"]],
            "human_order": human_order,
            "gold_order": LEVELS,
            "exact_correct": exact,
            "pairwise_correct": pair_correct,
        }
        path = annotation_path(username)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._json(
            {
                "levels": cid_to_level,
                "exact": exact,
                "pairwise_correct": pair_correct,
            }
        )

    def _handle_finish(self, body: dict) -> None:
        username = body.get("username", "")
        records = read_annotations(username)
        agreement = compute_agreement(records)
        summary = {
            "username": username,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "agreement": agreement,
        }
        saved_to = ""
        if records:
            summary_path = ANNOTATIONS_DIR / f"{slug(username)}__summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            saved_to = str(summary_path)
        self._json({"agreement": agreement, "saved_to": saved_to})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/v1_smoke_large",
        help="Generation output dir containing prompts.json (default: data/v1_smoke_large)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open a browser")
    args = parser.parse_args()

    data_dir = Path(args.data)
    global TRIPLETS, TRIPLET_BY_ID, ANNOTATIONS_DIR
    TRIPLETS = build_triplets(data_dir)
    TRIPLET_BY_ID = {t["task_id"]: t for t in TRIPLETS}
    ANNOTATIONS_DIR = data_dir / "annotations"
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRIPLETS:
        raise SystemExit(
            f"No complete triplets in {data_dir}/prompts.json "
            "(need a scenario with accepted prompts at all three levels)."
        )

    url = f"http://localhost:{args.port}/"
    n_pol = sum(1 for t in TRIPLETS if t["trait"] == "politeness")
    n_hed = len(TRIPLETS) - n_pol
    print(f"Loaded {len(TRIPLETS)} triplets ({n_pol} politeness, {n_hed} hedging).")
    print(f"Annotations will be saved under {ANNOTATIONS_DIR}/")
    print(f"Serving at {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    server = ThreadingHTTPServer(("localhost", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
