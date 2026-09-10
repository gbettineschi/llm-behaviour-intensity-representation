"""Blind triplet-ranking web app (library).

`human_eval(data_root)` serves a local web app and blocks until the annotator finishes:
they first pick one of the available datasets (a `data/<timestamp>/sentences/<trait>/` folder
with a `sentences_filtered.jsonl`), then order each scenario's three paraphrases — shown
unlabeled, in random order — from least to most of the trait. The true order is revealed after
each answer. On finish the server stops and the collected results are RETURNED to the caller,
which decides where to save them (see evaluate_sentences.py). Stdlib only — no extra
dependencies beyond the pure-data trait registry in lib.traits.
"""

from __future__ import annotations

import json
import random
import threading
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lib.traits import LEVELS, TRAITS  # LEVELS is the gold order, least -> most


# --- dataset -> blind triplets


def build_triplets(run_dir: Path) -> list[dict]:
    """One triplet per scenario that has a paraphrase at all three levels.

    Display order is shuffled (seeded per scenario, stable across reloads); the true level of
    each card is kept server-side for scoring.
    """
    path = run_dir / "sentences_filtered.jsonl"
    if not path.exists():
        return []
    by_scenario: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            by_scenario[r["scenario_id"]][r["level"]].append(r)

    triplets: list[dict] = []
    for sid, by_level in sorted(by_scenario.items()):
        if not all(by_level.get(lvl) for lvl in LEVELS):
            continue
        trait = by_level[LEVELS[0]][0]["trait"]
        chosen = {
            lvl: min(by_level[lvl], key=lambda r: r["text"])["text"] for lvl in LEVELS
        }
        display = LEVELS[:]
        random.Random(sid).shuffle(display)
        triplets.append(
            {
                "task_id": sid,
                "trait": trait,
                "axis": TRAITS.get(trait, {}).get("axis", f"least {trait}  →  most {trait}"),
                "cards": [
                    {"cid": f"c{i}", "level": lvl, "text": chosen[lvl]}
                    for i, lvl in enumerate(display)
                ],
            }
        )
    random.Random(20260521).shuffle(triplets)
    return triplets


# --- scoring


def score_order(human_order: list[str], assumed_order: list[str]) -> tuple[bool, int]:
    """Return (exact-match, #correctly-ordered-pairs) for one triplet."""
    rank = {lvl: i for i, lvl in enumerate(human_order)}
    assumed = {lvl: i for i, lvl in enumerate(assumed_order)}
    pairs = sum(
        1
        for a, b in combinations(assumed_order, 2)
        if (rank[a] < rank[b]) == (assumed[a] < assumed[b])
    )
    return human_order == assumed_order, pairs


def agreement(records: list[dict]) -> dict:
    """Aggregate agreement of the annotator's orderings against the dataset labels.

    For 3 levels Kendall tau is just 2*pairwise_accuracy - 1, so only the two independent
    numbers are reported: exact-order and pairwise accuracy.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    n_pairs = len(LEVELS) * (len(LEVELS) - 1) // 2
    exact = sum(r["exact"] for r in records)
    pairs = sum(r["pairwise_correct"] for r in records)
    return {
        "n": n,
        "exact_order_accuracy": round(exact / n, 4),
        "pairwise_accuracy": round(pairs / (n * n_pairs), 4),
    }


# --- web app

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human eval — blind ranking</title>
<style>
  :root { font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  body { margin: 0; background: #f4f4f6; color: #1c1c1e; }
  .wrap { max-width: 720px; margin: 0 auto; padding: 24px 18px 60px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #6b6b70; font-size: 13px; margin-bottom: 20px; }
  .panel { background: #fff; border: 1px solid #e3e3e6; border-radius: 12px; padding: 20px; }
  .ds { display: flex; gap: 10px; align-items: center; padding: 8px 4px; font-size: 14px; cursor: pointer; }
  .error { margin-top: 10px; font-size: 12px; color: #9c2a24; }
  button { font-size: 14px; padding: 9px 16px; border-radius: 8px; border: 1px solid #c8c8cc;
           background: #fff; cursor: pointer; }
  button.primary { background: #2b6cff; color: #fff; border-color: #2b6cff; }
  button:disabled { opacity: .45; cursor: default; }
  .progress { font-size: 13px; color: #6b6b70; margin-bottom: 12px; }
  .axis { font-size: 14px; margin: 6px 0 16px; }
  .axis b { color: #2b6cff; }
  .card { display: flex; gap: 12px; align-items: flex-start; background: #fff; cursor: pointer;
          border: 1px solid #e3e3e6; border-radius: 10px; padding: 14px; margin-bottom: 10px; }
  .card.ranked { border-color: #2b6cff; }
  .card.good { border-color: #1f9d55; background: #f1faf3; }
  .card.bad { border-color: #c0392b; background: #fdf2f1; }
  .badge { min-width: 64px; font-size: 12px; font-weight: 600; color: #2b6cff; }
  .text { font-size: 15px; line-height: 1.4; }
  .tag { font-size: 11px; color: #6b6b70; margin-top: 6px; }
  .row { display: flex; gap: 10px; margin-top: 12px; }
  .verdict { margin: 14px 0; font-size: 14px; padding: 10px 12px; border-radius: 8px; }
  .verdict.ok { background: #f1faf3; color: #1f7a44; }
  .verdict.no { background: #fdf2f1; color: #a23026; }
  table { border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #ececef; }
  .big { font-size: 30px; font-weight: 700; margin: 4px 0; }
  .hidden { display: none; }
</style></head>
<body><div class="wrap">
  <h1>Blind ranking</h1>
  <div class="sub">Order each scenario's three sentences from least to most of the trait. Labels are hidden.</div>

  <div id="pick" class="panel">
    <div class="sub">Pick a prompt dataset:</div>
    <div id="datasets"></div>
    <div class="row"><button id="startBtn" class="primary" disabled>Start</button></div>
    <div id="pickErr" class="error"></div>
  </div>

  <div id="task" class="panel hidden">
    <div id="progress" class="progress"></div>
    <div class="axis">Order: <b id="axis"></b></div>
    <div id="cards"></div>
    <div id="verdict" class="verdict hidden"></div>
    <div class="row">
      <button id="submitBtn" class="primary" disabled>Submit</button>
      <button id="resetBtn">Reset</button>
      <button id="nextBtn" class="primary hidden">Next</button>
    </div>
  </div>

  <div id="done" class="panel hidden">
    <h1>Results</h1>
    <div id="results"></div>
  </div>
</div>
<script>
const $ = (id) => document.getElementById(id);
let dataset = "", tasks = [], idx = 0, current = null, ranking = [], revealed = false;

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  return (await fetch(path, opt)).json();
}
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (100 * x).toFixed(1) + "%";

async function initPick() {
  const res = await api("/api/datasets");
  const box = $("datasets");
  const list = res.datasets || [];
  if (list.length === 0) { $("pickErr").textContent = "No datasets found under data/ (run generate_sentences.py first)."; return; }
  box.innerHTML = "";
  list.forEach((d, i) => {
    const row = document.createElement("label");
    row.className = "ds";
    row.innerHTML = '<input type="radio" name="ds" value="' + escapeHtml(d.name) + '">' +
                    '<span>' + escapeHtml(d.name) + ' — ' + d.n_triplets + ' triplets</span>';
    box.appendChild(row);
  });
  box.querySelectorAll('input[name="ds"]').forEach((el) => {
    el.onchange = () => { dataset = el.value; $("startBtn").disabled = false; };
  });
}

$("startBtn").onclick = async () => {
  if (!dataset) return;
  const res = await api("/api/tasks?dataset=" + encodeURIComponent(dataset));
  if (res.error) { $("pickErr").textContent = res.error; return; }
  tasks = res.tasks || [];
  $("pick").classList.add("hidden");
  if (tasks.length === 0) { showDone(); return; }
  $("task").classList.remove("hidden");
  idx = 0; renderTask();
};

function renderTask() {
  revealed = false; ranking = []; current = tasks[idx];
  $("progress").textContent = "Triplet " + (idx + 1) + " of " + tasks.length;
  $("axis").textContent = current.axis;
  $("verdict").classList.add("hidden");
  $("nextBtn").classList.add("hidden");
  $("submitBtn").classList.remove("hidden");
  $("resetBtn").classList.remove("hidden");
  $("submitBtn").disabled = true;
  const box = $("cards"); box.innerHTML = "";
  for (const card of current.cards) {
    const el = document.createElement("div");
    el.className = "card"; el.dataset.cid = card.cid;
    el.innerHTML = '<div class="badge">&middot;</div><div class="text">' +
                   escapeHtml(card.text) + '<div class="tag"></div></div>';
    el.onclick = () => clickCard(card.cid);
    box.appendChild(el);
  }
}

function clickCard(cid) {
  if (revealed || ranking.includes(cid)) return;
  ranking.push(cid); paintRanks();
  $("submitBtn").disabled = ranking.length !== 3;
}
function paintRanks() {
  const labels = { 0: "1 · least", 1: "2", 2: "3 · most" };
  for (const el of document.querySelectorAll(".card")) {
    const pos = ranking.indexOf(el.dataset.cid);
    el.classList.toggle("ranked", pos !== -1);
    el.querySelector(".badge").textContent = pos === -1 ? "·" : labels[pos];
  }
}
$("resetBtn").onclick = () => { if (!revealed) { ranking = []; paintRanks(); $("submitBtn").disabled = true; } };

$("submitBtn").onclick = async () => {
  if (ranking.length !== 3) return;
  const res = await api("/api/answer", { task_id: current.task_id, ranking });
  if (res.error) { $("verdict").textContent = res.error; return; }
  revealed = true;
  $("submitBtn").classList.add("hidden");
  $("resetBtn").classList.add("hidden");
  const assumedPos = {}; res.assumed_order.forEach((lvl, i) => { assumedPos[lvl] = i; });
  for (const el of document.querySelectorAll(".card")) {
    const level = res.levels[el.dataset.cid];
    el.classList.add(ranking.indexOf(el.dataset.cid) === assumedPos[level] ? "good" : "bad");
    el.querySelector(".tag").textContent = "true level: " + level.toUpperCase();
  }
  const v = $("verdict"); v.classList.remove("hidden");
  if (res.exact) { v.className = "verdict ok"; v.textContent = "Correct — matches the dataset order."; }
  else { v.className = "verdict no"; v.textContent = "Not quite — " + res.pairwise_correct + "/3 pairs right."; }
  $("nextBtn").classList.remove("hidden");
  if (idx + 1 >= tasks.length) $("nextBtn").textContent = "See results";
};

$("nextBtn").onclick = () => { idx += 1; (idx >= tasks.length) ? showDone() : renderTask(); };

async function showDone() {
  $("task").classList.add("hidden"); $("pick").classList.add("hidden");
  const res = await api("/api/finish", {});
  const a = res.agreement; const box = $("results");
  if (!a || !a.n) { box.innerHTML = "<p>No answers recorded.</p>"; }
  else {
    box.innerHTML =
      '<p class="big">' + pct(a.exact_order_accuracy) + '</p>' +
      '<p class="sub">exact-order agreement over ' + a.n + ' triplets</p>' +
      '<table><tr><th>n</th><th>exact order</th><th>pairwise</th></tr>' +
      '<tr><td>' + a.n + '</td><td>' + pct(a.exact_order_accuracy) + '</td><td>' +
      pct(a.pairwise_accuracy) + '</td></tr></table>' +
      '<p class="sub" style="margin-top:14px">Results returned to the launcher — you can close this tab.</p>';
  }
  $("done").classList.remove("hidden");
}
initPick();
</script></body></html>
"""


def _make_handler(data_root: Path, state: dict, results: dict, done: threading.Event):
    assumed_order = LEVELS

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _json(self, obj: dict, code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/datasets":
                datasets = []
                for p in sorted(data_root.glob("*/sentences/*/sentences_filtered.jsonl")):
                    datasets.append(
                        {
                            "name": f"{p.parent.parent.parent.name}/{p.parent.name}",  # <ts>/<trait>
                            "n_triplets": len(build_triplets(p.parent)),
                        }
                    )
                self._json({"datasets": datasets})
            elif parsed.path == "/api/tasks":
                name = parse_qs(parsed.query).get("dataset", [""])[0]
                ts, _, trait = name.partition("/")
                triplets = build_triplets(data_root / ts / "sentences" / trait)
                state.update(
                    dataset=name,
                    triplets=triplets,
                    by_id={t["task_id"]: t for t in triplets},
                    answers=[],
                )
                tasks = [
                    {
                        "task_id": t["task_id"],
                        "axis": t["axis"],
                        "cards": [
                            {"cid": c["cid"], "text": c["text"]} for c in t["cards"]
                        ],
                    }  # blind
                    for t in triplets
                ]
                self._json({"tasks": tasks})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path == "/api/answer":
                self._answer(self._body())
            elif self.path == "/api/finish":
                self._finish()
            else:
                self._json({"error": "not found"}, 404)

        def _answer(self, body: dict) -> None:
            triplet = state["by_id"].get(body.get("task_id", ""))
            ranking = body.get("ranking", [])
            cid_to_level = (
                {c["cid"]: c["level"] for c in triplet["cards"]} if triplet else {}
            )
            if triplet is None or set(ranking) != set(cid_to_level):
                self._json({"error": "invalid answer"}, 400)
                return
            human_order = [cid_to_level[cid] for cid in ranking]
            exact, pairs = score_order(human_order, assumed_order)
            state["answers"].append(
                {
                    "scenario_id": triplet["task_id"],
                    "trait": triplet["trait"],
                    "human_order": human_order,
                    "exact": exact,
                    "pairwise_correct": pairs,
                }
            )
            self._json(
                {
                    "levels": cid_to_level,
                    "assumed_order": assumed_order,
                    "exact": exact,
                    "pairwise_correct": pairs,
                }
            )

        def _finish(self) -> None:
            results.update(
                dataset=state["dataset"],
                assumed_order=list(LEVELS),
                answers=state["answers"],
                agreement=agreement(state["answers"]),
            )
            self._json({"agreement": results["agreement"]})
            done.set()

    return Handler


def human_eval(data_root, *, port: int = 8765, open_browser: bool = True) -> dict:
    """Serve the blind-ranking app over data_root and block until the annotator finishes.

    Returns {dataset, answers, agreement}; the caller decides where to save it.
    """
    data_root = Path(data_root).expanduser().resolve()
    results: dict = {}
    done = threading.Event()
    state = {"dataset": None, "triplets": [], "by_id": {}, "answers": []}
    server = ThreadingHTTPServer(
        ("localhost", port), _make_handler(data_root, state, results, done)
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}/"
    print(
        f"Serving at {url}  — pick a dataset, rate, and finish in the browser. (Ctrl-C to abort)"
    )
    if open_browser:
        webbrowser.open(url)
    try:
        done.wait()
    except KeyboardInterrupt:
        print("\nAborted.")
    server.shutdown()
    return results
