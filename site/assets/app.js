/* Main script of the ABench demo site. Zero dependencies: animation uses the Web Animations API, requestAnimationFrame and IntersectionObserver. */
(() => {
  "use strict";
  const A = window.ABENCH;
  if (!A) { document.body.insertAdjacentHTML("afterbegin", "<p style='padding:24px'>data/abench-data.js not found. Run python3 tools/build_site_data.py first.</p>"); return; }
  const IDX = A.index, C = A.case;
  const NS = "http://www.w3.org/2000/svg";
  const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ utilities
  const $ = (s, r = document) => r.querySelector(s);
  const esc = s => String(s ?? "").replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
  const fmt = (v, d = 2) => (v == null || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
  const ease = t => 1 - Math.pow(1 - t, 3);
  const sleep = ms => new Promise(r => setTimeout(r, RM ? 0 : ms));
  function S(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "css") n.setAttribute("style", v);
      else if (k === "text") n.textContent = v;
      else n.setAttribute(k, v);
    }
    if (parent) parent.appendChild(n);
    return n;
  }
  function onView(node, cb, threshold = 0.25) {
    if (!("IntersectionObserver" in window)) { cb(); return; }
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); cb(); } }, { threshold });
    io.observe(node);
  }
  function tween(dur, fn, ez = ease) {
    return new Promise(res => {
      if (RM || dur <= 0) { fn(1); res(); return; }
      const t0 = performance.now();
      const step = now => { const t = Math.min(1, (now - t0) / dur); fn(ez(t)); t < 1 ? requestAnimationFrame(step) : res(); };
      requestAnimationFrame(step);
    });
  }
  function drawIn(path, dur = 600, delay = 0) {
    if (RM) return;
    let len = 0;
    try { len = path.getTotalLength(); } catch (e) { return; }
    if (!len) return;
    const dash = path.getAttribute("stroke-dasharray");
    path.style.strokeDasharray = `${len} ${len}`;
    path.style.strokeDashoffset = len;
    const a = path.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: dur, delay, easing: "cubic-bezier(.3,.6,.2,1)", fill: "forwards" });
    a.onfinish = () => { path.style.strokeDasharray = dash || ""; path.style.strokeDashoffset = ""; a.cancel(); };
  }
  function pop(node, delay = 0, dur = 380) {
    if (RM) return;
    node.style.transformBox = "fill-box";
    node.style.transformOrigin = "center";
    node.animate([{ opacity: 0, transform: "scale(.4)" }, { opacity: 1, transform: "scale(1)" }],
      { duration: dur, delay, easing: "cubic-bezier(.2,1.4,.4,1)", fill: "backwards" });
  }
  function fade(node, delay = 0, dur = 400, dy = 6) {
    if (RM) return;
    node.animate([{ opacity: 0, transform: `translateY(${dy}px)` }, { opacity: 1, transform: "none" }],
      { duration: dur, delay, easing: "ease-out", fill: "backwards" });
  }
  function countTo(node, to, dur = 900, d = 2, suffix = "") {
    tween(dur, t => { node.textContent = (to * t).toFixed(d) + suffix; });
  }
  const checkIcon = ok => ok
    ? `<svg class="mark" viewBox="0 0 22 22" aria-label="pass"><circle cx="11" cy="11" r="10" fill="none" style="stroke:var(--green)" stroke-width="2"/><path d="M6 11.5l3.2 3.2L16 8" fill="none" style="stroke:var(--green)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>`
    : `<svg class="mark" viewBox="0 0 22 22" aria-label="fail"><circle cx="11" cy="11" r="10" fill="none" style="stroke:var(--red)" stroke-width="2"/><path d="M7 7l8 8M15 7l-8 8" style="stroke:var(--red)" stroke-width="2.2" stroke-linecap="round"/></svg>`;

  // ------------------------------------------------------------------ names and derived data
  const PAPER_SHORT = { adam: "Adam", adamw: "AdamW", adafactor: "Adafactor", shampoo: "Shampoo", lamb: "LAMB", crowded_valley: "Crowded Valley", lion: "Lion", sophia: "Sophia", schedulefree: "Schedule-Free", deconstructing: "Deconstructing", muon_blog: "Muon blog", ademamix: "AdEMAMix", soap: "SOAP", mars: "MARS", cautious: "Cautious", muon_scalable: "Muon at scale", kimi_k2: "Kimi K2", bench_optim: "Optimizer benchmark", fantastic: "Fantastic", rudelson_vershynin_2007: "Stable rank", vonmises1929: "Power iteration" };
  const REF_NAME = Object.fromEntries(C.references.map(r => [r.id, r.name.replace(/\s*\(.*\)/, "")]));
  REF_NAME.cautious_adamw = "C-AdamW"; REF_NAME.schedulefree_adamw = "Schedule-Free";
  const SETTING_NAME = { L2_main: "μ2 public", L3a_scale: "μ3 scale", L3b_long: "μ1 long training", L3c_bigbatch: "μ2 large batch" };
  const SETTING_DESC = { L2_main: "about 17M effective parameters, 3 seeds, results public", L3a_scale: "about 42M, hidden scale-extrapolation point", L3b_long: "about 7M with 8× data; checks whether a mid-training advantage survives learning-rate decay", L3c_bigbatch: "about 17M with 4× batch size" };
  const GROUP_TITLE = { method: "Method tracks", frontier: "Frontier exploration", domain: "Professional domains", pro: "Pro Set", entry: "Entry level" };
  const GROUP_ORDER = ["method", "frontier", "domain", "pro", "entry"];
  const LEVELS = ["low", "medium", "high"];
  const LEVEL_ZH = { low: "low", medium: "medium", high: "high", pro: "pro" };
  // ``status`` is the repository artifact state; ``readiness`` is the
  // evaluation state.  Keep scaffold visible so it never renders as
  // ``undefined`` or gets mistaken for a finished benchmark.
  const STATUS_ZH = { built: "built", spec: "spec final", scaffold: "skeleton built", planned: "scheduled" };
  const READINESS_ZH = { deferred: "deferred", contract_only: "contract only", proxy_ready: "proxy ready", reference_ready: "reference ready", hidden_ready: "hidden set ready" };
  const SUBSET_ZH = { main: "main", calibration: "calibration", primitive: "primitive search", stress: "stress" };
  const KIND_ZH = { code_only: "code", code_plus_checkpoint: "code + weights", report_only: "judgment report" };
  const KIND_READS = { code_only: "runs only the submitted code: it trains from scratch where training is what is measured (optimizers, time to target), otherwise it only runs and measures", code_plus_checkpoint: "loads only the submitted model code and the checkpoint trained in the sandbox, for inference and measurement; train.py is kept for audit and never run", report_only: "judges review the judgment report and minimal verification plan" };
  const MODE = {
    checkpoint: { name: "Checkpoint", tag: "green", what: "model code + checkpoint/ + train.py",
      does: "The agent trains in the sandbox and submits the checkpoint with the train.py that produced it. The evaluator only loads it, infers and measures; it never retrains and never runs train.py." },
    judge_trains: { name: "Evaluator trains", tag: "amber", what: "code",
      does: "The training program itself is what is measured (time to a target accuracy or loss, or an optimizer), so the evaluator trains from scratch on hidden seeds. The agent still develops with the same pipeline in the sandbox." },
    no_training: { name: "No training", tag: "plain", what: "code",
      does: "Kernels and samplers: the evaluator runs and measures the submitted code; nothing is trained." },
  };
  const JUDGE_ZH = { pending: "judge to be frozen", tabular_classification: "judge wired in",
                     timed_train_classification: "timed judge wired in", harness: "stage-pipeline judge" };
  const allCases = IDX.tracks.flatMap(t => t.cases.map(c => ({ ...c, track: t, group: t.group })));
  const papers = C.papers;
  const paperById = Object.fromEntries(papers.map(p => [p.id, p]));
  function resolveSrc(sid) {
    const s = String(sid).replace(/^arxiv:\s*/i, "");
    return papers.find(p => p.id === s || p.arxiv === s) || null;
  }
  const fy = d => { const [y, m, dd] = d.split("-").map(Number); return y + (m - 1) / 12 + (dd - 1) / 365; };

  // ------------------------------------------------------------------ theme and navigation
  $("#themeToggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("abench-theme", root.dataset.theme); } catch (e) {}
  });
  (() => {
    const links = [...document.querySelectorAll(".nav a")];
    const map = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) { links.forEach(l => l.classList.remove("on")); map.get(e.target.id)?.classList.add("on"); } });
    }, { rootMargin: "-45% 0px -50% 0px" });
    map.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
  })();

  // ------------------------------------------------------------------ hero
  (() => {
    const facts = [[IDX.tracks.length, "tracks"], [allCases.length, "cases"], [allCases.filter(c => c.oj?.judge && c.oj.judge !== "pending").length, "judges wired in"], [allCases.filter(c => c.oj?.standard).length, "standard cases"]];
    $("#heroFacts").innerHTML = facts.map(([n, l]) => `<div><dt>${l}</dt><dd>${n}</dd></div>`).join("");
    $("#heroCasesBtn").textContent = `Browse all ${allCases.length} cases`;

    const svg = $("#heroSvg");
    const W = 1200, X0 = 70, X1 = 1000;
    // time axis: take the log of (2027.2 − year), compressing early years and expanding recent ones
    const T = y => -Math.log(2027.2 - y);
    const xOf = y => X0 + (T(y) - T(2014)) / (T(2026.75) - T(2014)) * (X1 - X0);
    const cutoffX = xOf(fy("2026-07-01"));
    const ml = papers.filter(p => p.field === "ml").sort((a, b) => a.date.localeCompare(b.date));
    const ext = papers.filter(p => p.field !== "ml").sort((a, b) => a.date.localeCompare(b.date));
    const FS = 13;
    const textW = s => [...s].reduce((a, ch) => a + (/[\u4e00-\u9fff]/.test(ch) ? FS : FS * 0.58), 0);
    const ROWS = [250, 210, 290, 170, 330, 130, 370, 90, 410];
    const boxes = [];
    const overlap = (a, b) => !(a.x2 < b.x1 || b.x2 < a.x1 || a.y2 < b.y1 || b.y2 < a.y1);
    ml.forEach(p => {
      p._x = xOf(fy(p.date));
      p._r = p.role === "anchor" ? 16 : 5 + 12 * (p.w ?? 0.4);
      p._label = PAPER_SHORT[p.id] || p.id;
      if (p.role === "anchor") {
        p._y = 250;
        boxes.push({ x1: p._x - 120, x2: p._x + p._r, y1: p._y - 22, y2: p._y + 30 });
        return;
      }
      let pick = ROWS[0], best = Infinity;
      for (const y of ROWS) {
        const b = { x1: p._x - p._r - 2, x2: p._x + p._r + 6 + textW(p._label), y1: y - 12, y2: y + 12 };
        const hits = boxes.filter(o => overlap(o, b)).length;
        const cost = hits * 1000 + Math.abs(y - 250) * 0.2;
        if (cost < best) { best = cost; pick = y; }
        if (hits === 0) break;
      }
      p._y = pick;
      boxes.push({ x1: p._x - p._r - 2, x2: p._x + p._r + 6 + textW(p._label), y1: pick - 12, y2: pick + 12 });
    });
    const EXT_Y = 494;
    ext.forEach((p, i) => { p._x = 210 + i * 210; p._y = EXT_Y; p._r = 5 + 12 * (p.w ?? 0.4); p._label = `${PAPER_SHORT[p.id] || p.id} (${p.venue.replace(/\s*\d{4}$/, "")}, ${p.year})`; });

    const ex = C.submissions.example_submission;
    const srcRows = ex.Q.sources;
    const prop = { x: 1112, y: 250 };

    function build() {
      svg.innerHTML = "";
      const gAxis = S("g", {}, svg), gEdges = S("g", {}, svg), gProp = S("g", {}, svg), gNodes = S("g", {}, svg), gTop = S("g", {}, svg);
      const axisY = 446;
      S("line", { x1: X0 - 10, x2: X1 + 20, y1: axisY, y2: axisY, class: "axis" }, gAxis);
      [2014, 2018, 2020, 2022, 2023, 2024, 2025, 2026].forEach(y => {
        const x = xOf(y);
        S("line", { x1: x, x2: x, y1: axisY, y2: axisY + 6, class: "axis" }, gAxis);
        S("text", { x, y: axisY + 22, "text-anchor": "middle", "font-size": 12, class: "t-mono", text: y }, gAxis);
      });
      S("text", { x: X0 - 10, y: 36, "font-size": FS, text: "ML neighborhood of B-OPT-001 · node size = paper weight w; the time axis expands toward recent years" }, gAxis);
      S("line", { x1: X0 - 10, x2: W - 20, y1: EXT_Y - 26, y2: EXT_Y - 26, class: "grid" }, gAxis);
      S("text", { x: X0 - 10, y: EXT_Y + 4, "font-size": 12.5, text: "cross-field sources" }, gAxis);

      const anchor = ml.find(p => p.role === "anchor");
      const nbEdges = [];
      ml.forEach(p => {
        if (p === anchor) return;
        const mx = (p._x + anchor._x) / 2, my = (p._y + anchor._y) / 2 - 18;
        nbEdges.push([p, S("path", { d: `M${anchor._x},${anchor._y} Q${mx},${my} ${p._x},${p._y}`, fill: "none", css: "stroke:var(--ink-3)", "stroke-width": 1, opacity: 0.28 }, gEdges)]);
      });

      const nodeEls = [];
      [...ml, ...ext].forEach(p => {
        const g = S("g", {}, gNodes);
        S("title", { text: `${p.title} (${p.venue}, ${p.date})  w = ${p.w}` }, g);
        let style;
        if (p.role === "anchor") style = "fill:var(--ink);stroke:var(--ink)";
        else if (p.role === "reference") style = "fill:var(--blue-soft);stroke:var(--blue)";
        else if (p.role === "T7") style = "fill:var(--sheet);stroke:var(--amber)";
        else if (p.role === "external") style = "fill:var(--green-soft);stroke:var(--green)";
        else style = "fill:var(--sheet);stroke:var(--ink-2)";
        S("circle", { cx: p._x, cy: p._y, r: p._r, css: style, "stroke-width": 1.5, "stroke-dasharray": p.role === "T7" ? "3 2" : null }, g);
        if (p.role === "anchor") {
          S("text", { x: p._x - p._r - 8, y: p._y + 1, "text-anchor": "end", "font-size": 17, "font-weight": 700, class: "t-ink", text: "Adam" }, g);
          S("text", { x: p._x - p._r - 8, y: p._y + 19, "text-anchor": "end", "font-size": 12, text: "anchor · 2014" }, g);
        } else {
          S("text", { x: p._x + p._r + 5, y: p._y + 4.5, "font-size": FS, text: p._label }, g);
        }
        nodeEls.push([p, g]);
      });

      const cut = S("path", { d: `M${cutoffX},50 L${cutoffX},${axisY}`, fill: "none", css: "stroke:var(--red)", "stroke-width": 1.8, "stroke-dasharray": "6 4" }, gTop);
      const cutLabel = S("g", {}, gTop);
      S("text", { x: cutoffX + 8, y: axisY - 26, "font-size": 12.5, class: "t-red", text: "cutoff 2026-07-01" }, cutLabel);
      S("text", { x: cutoffX + 8, y: axisY - 10, "font-size": 12, class: "t-red", text: "not retrievable after" }, cutLabel);

      const propEdges = [];
      let ghost = null;
      srcRows.forEach(s => {
        const p = resolveSrc(s.id);
        let tx, ty, style = "stroke:var(--blue)", w = 1.3, dash = null, op = 0.9;
        if (!p) { tx = xOf(2026.37); ty = 58; style = "stroke:var(--red)"; dash = "4 3"; ghost = { x: tx, y: ty, id: s.id }; }
        else {
          tx = p._x; ty = p._y;
          if (s.usage === "core") { w = 2.4; }
          else if (s.usage === "partial") { w = 1.2; op = 0.65; }
          else { style = "stroke:var(--ink-3)"; dash = "3 4"; op = 0.8; }
        }
        const sx = prop.x - 60, sy = prop.y + Math.max(-20, Math.min(20, (ty - prop.y) * 0.15));
        let d;
        if (ty > 460) {
          d = `M${prop.x - 20},${prop.y + 26} C${prop.x - 20},${ty} ${tx + 260},${ty} ${tx + p._r},${ty}`;
        } else {
          const mx = (sx + tx) / 2;
          const my = Math.abs(ty - 250) < 5 ? 48 : ty < 250 ? Math.min(sy, ty) - 46 : Math.max(sy, ty) + 46;
          d = `M${sx},${sy} Q${mx},${my} ${tx},${ty}`;
        }
        propEdges.push(S("path", { d, fill: "none", css: style, "stroke-width": w, "stroke-dasharray": dash, opacity: op }, gProp));
      });
      let ghostG = null;
      if (ghost) {
        ghostG = S("g", {}, gProp);
        S("title", { text: `${ghost.id}: unresolvable source, reference quality × 0.8` }, ghostG);
        S("circle", { cx: ghost.x, cy: ghost.y, r: 10, css: "fill:var(--red-soft);stroke:var(--red)", "stroke-width": 1.5, "stroke-dasharray": "3 2" }, ghostG);
        S("path", { d: `M${ghost.x - 4},${ghost.y - 4} L${ghost.x + 4},${ghost.y + 4} M${ghost.x + 4},${ghost.y - 4} L${ghost.x - 4},${ghost.y + 4}`, css: "stroke:var(--red)", "stroke-width": 1.7 }, ghostG);
        S("text", { x: ghost.x - 16, y: ghost.y - 4, "text-anchor": "end", "font-size": 12.5, class: "t-red", text: "declared source " + ghost.id }, ghostG);
        S("text", { x: ghost.x - 16, y: ghost.y + 12, "text-anchor": "end", "font-size": 12, class: "t-red", text: "unresolvable, Q × 0.8" }, ghostG);
      }
      const pg = S("g", {}, gProp);
      S("rect", { x: prop.x - 60, y: prop.y - 26, width: 120, height: 52, rx: 6, css: "fill:var(--sheet);stroke:var(--ink)", "stroke-width": 1.8 }, pg);
      S("text", { x: prop.x, y: prop.y - 3, "text-anchor": "middle", "font-size": 16, "font-weight": 700, class: "t-ink", text: "RankGate" }, pg);
      S("text", { x: prop.x, y: prop.y + 15, "text-anchor": "middle", "font-size": 12, text: "agent's method" }, pg);
      const ring = S("path", { d: `M${prop.x - 74},${prop.y + 6} C${prop.x - 78},${prop.y - 46} ${prop.x + 70},${prop.y - 52} ${prop.x + 78},${prop.y - 6} C${prop.x + 84},${prop.y + 42} ${prop.x - 44},${prop.y + 52} ${prop.x - 70},${prop.y + 22}`, fill: "none", css: "stroke:var(--red)", "stroke-width": 2, "stroke-linecap": "round" }, pg);
      const note = S("g", {}, pg);
      S("text", { x: prop.x + 8, y: prop.y - 76, "text-anchor": "middle", "font-size": 13, class: "t-red", text: "overall " + fmt(ex.composite.score, 2) }, note);
      S("text", { x: prop.x + 8, y: prop.y - 60, "text-anchor": "middle", "font-size": 12, class: "t-red", text: "incl. local-discovery bonus" }, note);
      return { nbEdges, nodeEls, cut, cutLabel, propEdges, ghostG, pg, ring, note, gAxis };
    }

    let token = 0;
    function play() {
      token++;
      const r = build();
      if (RM) return;
      fade(r.gAxis, 0, 500, 0);
      r.nodeEls.forEach(([p, g], i) => {
        const d = 200 + i * 110;
        pop(g, d);
        const e = r.nbEdges.find(x => x[0] === p);
        if (e) drawIn(e[1], 500, d);
      });
      const t1 = 200 + r.nodeEls.length * 110 + 150;
      drawIn(r.cut, 520, t1);
      fade(r.cutLabel, t1 + 300, 400, 0);
      pop(r.pg, t1 + 500, 500);
      r.propEdges.forEach((e, i) => drawIn(e, 650, t1 + 800 + i * 130));
      if (r.ghostG) pop(r.ghostG, t1 + 800 + r.propEdges.length * 130);
      drawIn(r.ring, 800, t1 + 2100);
      fade(r.note, t1 + 2600, 400, 0);
    }
    const legend = [
      ["anchor", "fill:var(--ink);stroke:var(--ink)"], ["reference method", "fill:var(--blue-soft);stroke:var(--blue)"],
      ["other neighborhood", "fill:var(--sheet);stroke:var(--ink-2)"], ["T7 analysis and negative results", "fill:var(--sheet);stroke:var(--amber);stroke-dasharray:3 2"],
      ["cross-field source", "fill:var(--green-soft);stroke:var(--green)"]
    ];
    $("#heroLegend").innerHTML = legend.map(([l, s]) => `<span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5.5" style="${s}" stroke-width="1.4"/></svg>${l}</span>`).join("")
      + `<span><svg viewBox="0 0 14 14"><path d="M1 7h12" style="stroke:var(--blue)" stroke-width="2.4"/></svg>core source of the method</span>`
      + `<span><svg viewBox="0 0 14 14"><path d="M1 7h12" style="stroke:var(--blue);opacity:.65" stroke-width="1.2"/></svg>partly borrowed</span>`
      + `<span><svg viewBox="0 0 14 14"><path d="M1 7h12" style="stroke:var(--ink-3)" stroke-width="1.4" stroke-dasharray="3 3"/></svg>decorative citation</span>`;
    onView(svg, play, 0.2);
    $("#heroReplay").addEventListener("click", play);
    void token;
  })();

  // ------------------------------------------------------------------ build flow S0–S6
  (() => {
    const ex = C.submissions.example_submission;
    const lad = C.ladder;
    const STEPS = [
      { code: "S0", name: "Issue view", title: "Issue: a read-only agent view", text: "The agent gets the task statement, interface notes, report templates and public training data (none for this task). Hidden settings, the reference pool, the hidden set and the scoring weights stay outside the sandbox; the view is checked for leaked judge-side material when it is built.", vis: visView },
      { code: "S1", name: "Open retrieval", title: "Retrieve: any paper from before the cutoff", text: "The retrieval backend blocks papers after the cutoff and returns no citation counts; calls are logged by the operator. Below is the historical retrieval log of the example submission.", vis: visLog },
      { code: "S2", name: "Synthesize", title: "Synthesize: lineage, evidence and gaps", text: "Organize the neighborhood into a technical lineage, separate strong from weak evidence, and find what existing methods do not cover.", vis: visLineage },
      { code: "S3", name: "Design", title: "Design: the RankGate update rule", text: "Interpolates continuously between the orthogonalized and factored adaptive updates by the normalized stable rank of the momentum matrix. The gate works on a log scale with threshold τ = 0.02.", vis: visDesign },
      { code: "S3′", name: "Self-test", title: "Self-test: trial and error in its own workspace", text: "With a development budget of 8 GPU-h, the agent trains and judges for itself in a writable workspace. Here it found that the linear gate was almost always 0 and switched to a log gate. The agent cannot see the evaluation settings.", vis: visBudget },
      { code: "S4", name: "Design report", title: "Report: arXiv layout, principles only", text: "Write report/index.html from the template: introduction, background, method (structure diagram, equations, algorithm, complexity), design rationale, scope; references registered in references.yaml. No experiments or numbers.", vis: visDoc },
      { code: "S5", name: "Submit", title: "Submit: one-shot delivery, allowlisted collection", text: "B-OPT-001 submits code only: the evaluation environment trains it from scratch. Collection keeps only allowlisted files and drops self-reported metrics; the agent's workspace is then deleted.", vis: visFiles },
      { code: "S6", name: "OJ evaluation", title: "Evaluate: outside the sandbox, reading code and weights", text: "The judge checks the format and then runs the submission on hidden settings. Tasks like DS-GBM-001 load the checkpoint directly and predict on the encrypted test set; the retraining judge of B-OPT-001 is defined, with data still to be frozen.", vis: visJudge },
      { code: "S7", name: "Score", title: "Score: thresholds and four components", text: "First pass three thresholds: interface check, staying at SOTA, differentiation. Then score SOTA margin, divergence and reference quality, and multiply by the local-discovery bonus.", vis: visMeters },
      { code: "S8", name: "Report", title: "Report: overall score and itemized details", text: "Lists per-setting speedups, the behavioral nearest neighbor, source contributions and the ablation of each discovery. The numbers below are historical simulations, used only to demonstrate the scoring logic.", vis: visReport },
    ];
    const ol = $("#steps"), stage = $("#stepStage");
    ol.style.setProperty("--n", STEPS.length);
    ol.innerHTML = STEPS.map((s, i) => `<li><button type="button" role="tab" aria-selected="false" data-i="${i}"><span class="dot-row"><span class="dot"></span><span class="rail"><i></i></span></span><span class="code">${s.code}</span><span class="name">${s.name}</span></button></li>`).join("");
    let cur = -1, timer = null;
    function show(i) {
      cur = i;
      [...ol.children].forEach((li, j) => { li.classList.toggle("on", j === i); li.classList.toggle("done", j < i); li.querySelector("button").setAttribute("aria-selected", j === i); });
      const s = STEPS[i];
      stage.innerHTML = `<div class="st-text"><span class="mono" style="color:var(--ink-3)">${s.code} · ${i < 6 ? "inside the sandbox" : "outside the sandbox"}</span><h3>${s.title}</h3><p>${s.text}</p></div><div class="st-vis"></div>`;
      fade(stage.querySelector(".st-text"), 0, 350);
      s.vis(stage.querySelector(".st-vis"));
    }
    ol.addEventListener("click", e => { const b = e.target.closest("button[data-i]"); if (b) { stop(); show(+b.dataset.i); } });
    ol.addEventListener("keydown", e => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault(); stop();
      const n = Math.max(0, Math.min(STEPS.length - 1, cur + (e.key === "ArrowRight" ? 1 : -1)));
      show(n); ol.children[n].querySelector("button").focus();
    });
    $("#stepPrev").addEventListener("click", () => { stop(); show(Math.max(0, cur - 1)); });
    $("#stepNext").addEventListener("click", () => { stop(); show(Math.min(STEPS.length - 1, cur + 1)); });
    const playBtn = $("#stepPlay");
    function stop() { if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "Autoplay"; } }
    playBtn.addEventListener("click", () => {
      if (timer) { stop(); return; }
      if (cur >= STEPS.length - 1) show(0);
      playBtn.textContent = "Pause";
      timer = setInterval(() => { if (cur >= STEPS.length - 1) { stop(); return; } show(cur + 1); }, 4600);
    });
    show(0);
    onView($("#flow"), () => { if (cur === 0 && !timer && !RM) playBtn.click(); }, 0.35);

    function visView(box) {
      const o = (allCases.find(c => c.id === "B-OPT-001") || {}).oj || {};
      const seen = (o.view || []).map(f => `<li class="f"><span class="mono">view/${esc(f)}</span></li>`).join("");
      const hidden = ["case/eval_protocol.yaml", "case/scoring_config.yaml", "case/metric_registry.yaml", "case/reference_pool.yaml", "case/anti_gaming.md", "harness/reference_rules.py", "case/oj.yaml"];
      box.innerHTML = `<div class="view-cmp"><div><span class="k">the agent can see</span><ul class="tree small">${seen}</ul>
        <div class="card-sheet" style="margin-top:10px">${[["Anchor", "Adam (arXiv 1412.6980)"], ["Cutoff", "2026-07-01"], ["Development budget", "8 GPU-h, one 24GB card"], ["Submissions", "one"]].map(r => `<div class="row"><span>${r[0]}</span><span>${esc(r[1])}</span></div>`).join("")}</div></div>
        <div><span class="k">cannot see</span><ul class="tree small gone">${hidden.map(f => `<li class="f"><span class="mono">${f}</span></li>`).join("")}</ul></div></div>`;
      [...box.querySelectorAll(".tree li, .row")].forEach((r, i) => fade(r, 60 + i * 60, 260, 4));
      [...box.querySelectorAll(".gone li")].forEach((r, i) => setTimeout(() => r.classList.add("x"), RM ? 0 : 900 + i * 90));
    }
    function visDoc(box) {
      const toc = ["Abstract", "1 Introduction", "2 Background", "3 Method", "3.1 Overall structure", "3.2 Definitions", "3.3 Algorithm", "3.4 Complexity", "4 Design Rationale", "5 Scope and Limitations", "References"];
      box.innerHTML = `<div class="mini-page"><div class="mp-bar"></div><div class="mp-body"><div class="mp-toc">${toc.map(x => `<i>${x}</i>`).join("")}</div>
        <div class="mp-main"><b class="mp-title">RankGate: An Optimizer that Adjusts Orthogonalization Strength by the Stable Rank of the Momentum</b><span class="mp-abs"></span><span class="mp-h">1 Introduction</span><span class="mp-p"></span><span class="mp-p s"></span>
        <span class="mp-h">3.2 Definitions</span><span class="mp-eq">r = ‖M̃‖²_F / (‖M̃‖²₂ · min(m,n))<em>(3)</em></span><span class="mp-eq">a = σ(κ(ln r − ln τ))<em>(3)</em></span>
        <span class="mp-alg">Algorithm 1  RankGate (one step for a single 2D parameter)</span><span class="mp-h">4 Design Rationale</span><span class="mp-thm">Proposition 1.  At rank one it reduces to the factored adaptive branch…</span><span class="mp-ref">[Kingma &amp; Ba 2015] [Jordan et al. 2024] [Rudelson &amp; Vershynin 2007] …</span></div></div></div>
        <p class="panel-note" style="margin-top:10px">Section headings containing words such as "Experiments", "Results", "Ablation" or "Evaluation" fail the check. <a href="#report">See the full report</a></p>`;
      [...box.querySelectorAll(".mp-toc i, .mp-main > *")].forEach((n, i) => fade(n, 60 + i * 70, 260, 3));
    }
    function visJudge(box) {
      const rows = [["Format check", "layout, allowlist, report structure, consistent references", "ok"], ["Decrypt hidden set", "the judge holds the key exclusively and decrypts into its own temporary directory", "ok"], ["Run the submission", "the subprocess gets only a copy of the submission and the features; minimal environment, no network, resource limits", "ok"],
        ["Compute metrics", "labels exist only in the parent process's memory", "ok"], ["B-OPT-001: retraining judge", "trains from scratch on hidden settings with a fixed small LM; data still to be frozen", "wait"]];
      box.innerHTML = `<div class="dr-flow">${rows.map(r => `<div class="fl ${r[2]}"><span class="d"></span><span class="c"><b>${r[0]}</b><span>${r[1]}</span></span></div>`).join("")}</div>
        <p class="panel-note" style="margin-top:8px">See the step-by-step animation in the "OJ evaluation" section below.</p>`;
      [...box.querySelectorAll(".fl")].forEach((r, i) => fade(r, 80 + i * 160, 300, 4));
    }
    function visCard(box) {
      const rows = [["Case", "B-OPT-001 · track B optimizers"], ["Anchor", "Adam (arXiv 1412.6980, ICLR 2015)"], ["Cutoff", "2026-07-01"], ["Goal", "Design a new optimizer for small-scale LLM pretraining: stay at SOTA and be distinguishable from existing methods"],
        ["Development budget", "8 GPU-h, one 24GB reference card"], ["Resources / difficulty", "high / medium"], ["Submissions", "one"]];
      box.innerHTML = `<div class="card-sheet">${rows.map(r => `<div class="row"><span>${r[0]}</span><span>${esc(r[1])}</span></div>`).join("")}</div>`;
      [...box.querySelectorAll(".row")].forEach((r, i) => fade(r, 80 + i * 90, 300, 4));
    }
    function visLog(box) {
      box.innerHTML = `<div class="log"></div>`;
      const log = box.firstChild;
      C.retrieval.forEach((e, i) => {
        const a = e.args || {};
        const arg = a.query ? `"${a.query}"` : a.id ? a.id : a.hparams ? JSON.stringify(a.hparams) : a.path || "";
        const res = e.result_ids?.length ? ` → ${e.result_ids.length} papers` : "";
        const ln = document.createElement("div");
        ln.className = "ln";
        ln.innerHTML = `<span class="tm">${e.t.slice(11, 16)}</span><span class="tool">${esc(e.tool)}</span><span class="arg">${esc(arg)}${res}</span>`;
        log.appendChild(ln);
        fade(ln, 150 + i * 260, 260, 4);
      });
    }
    function visLineage(box) {
      const lanes = [["Element-wise adaptive", ["Adam", "AdamW", "Adafactor", "LAMB"]], ["Sign and search", ["Lion"]], ["Second-order preconditioning", ["Shampoo", "SOAP", "Sophia"]],
        ["Orthogonalization", ["Muon", "Muon at scale", "MuonClip"]], ["Momentum structure", ["AdEMAMix", "MARS", "Cautious", "Schedule-Free"]], ["T7 evidence", ["Fantastic: speedup shrinks with scale", "Deconstructing"]]];
      box.innerHTML = `<div class="lineage">${lanes.map(l => `<div class="lane"><span>${l[0]}</span><span class="nodes">${l[1].map(n => `<span class="tag ${l[0].startsWith("T7") ? "amber" : ""}">${n}</span>`).join("")}</span></div>`).join("")}
        <div class="gap">Gap: orthogonalization whitens every 2D parameter at every stage with the same strength, without adjusting to the spectral structure of the momentum itself.</div></div>`;
      [...box.querySelectorAll(".tag")].forEach((t, i) => pop(t, 100 + i * 70, 300));
      fade(box.querySelector(".gap"), 1500, 500);
    }
    function visDesign(box) {
      const eqs = [["M ← μM + G,  M̃ = G + μM", "Nesterov momentum"], ["O = 0.2·√max(m,n) · NS₅(M̃)", "orthogonalized branch"], ["A = RMS-match( M̃ / √(V̂ᵣ·V̂꜀) )", "factored adaptive branch"],
        ["r = ‖M̃‖²_F / (‖M̃‖²₂ · min(m,n))", "normalized stable rank"], ["a = σ( κ · (ln r − ln τ) )", "gate"], ["W ← W − η · (a·O + (1−a)·A + λW)", "update"]];
      box.innerHTML = `<div class="eqg">${eqs.map(e => `<span class="f">${e[0]}</span><span class="c">${e[1]}</span>`).join("")}</div><svg class="gate-svg" viewBox="0 0 420 192" role="img" aria-label="Gate value a as a function of stable rank r"></svg>`;
      [...box.querySelectorAll(".eqg span")].forEach((e, i) => fade(e, 60 + Math.floor(i / 2) * 120, 260, 3));
      const svg = box.querySelector("svg");
      const lx = r => 40 + (Math.log10(r) - Math.log10(0.003)) / (Math.log10(0.3) - Math.log10(0.003)) * 360;
      const ly = a => 140 - a * 110;
      S("rect", { x: lx(0.01), y: 26, width: lx(0.05) - lx(0.01), height: 114, css: "fill:var(--blue-soft)" }, svg);
      S("text", { x: (lx(0.01) + lx(0.05)) / 2, y: 22, "text-anchor": "middle", "font-size": 11, text: "range of r measured at L1" }, svg);
      S("line", { x1: 40, x2: 400, y1: 140, y2: 140, class: "axis" }, svg);
      S("line", { x1: 40, x2: 40, y1: 30, y2: 140, class: "axis" }, svg);
      [0.003, 0.01, 0.03, 0.1, 0.3].forEach(r => S("text", { x: lx(r), y: 156, "text-anchor": "middle", "font-size": 10.5, class: "t-mono", text: r }, svg));
      S("text", { x: 36, y: ly(1) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono", text: "1" }, svg);
      S("text", { x: 36, y: ly(0) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono", text: "0" }, svg);
      const curve = (fn) => { let d = ""; for (let i = 0; i <= 120; i++) { const lr = Math.log10(0.003) + i / 120 * (Math.log10(0.3) - Math.log10(0.003)); const r = 10 ** lr; d += (i ? "L" : "M") + lx(r).toFixed(1) + "," + ly(fn(r)).toFixed(1); } return d; };
      const sig = x => 1 / (1 + Math.exp(-x));
      const linear = S("path", { d: curve(r => sig(20 * (r - 0.15))), fill: "none", css: "stroke:var(--red)", "stroke-width": 1.6, "stroke-dasharray": "4 3" }, svg);
      const logg = S("path", { d: curve(r => sig(3 * (Math.log(r) - Math.log(0.02)))), fill: "none", css: "stroke:var(--blue)", "stroke-width": 2.4 }, svg);
      S("path", { d: "M40,184 L60,184", css: "stroke:var(--red)", "stroke-width": 1.6, "stroke-dasharray": "4 3" }, svg);
      S("text", { x: 66, y: 188, "font-size": 11, class: "t-red", text: "early linear gate τ=0.15: almost 0 over the measured range, so the method degenerates to Adafactor" }, svg);
      S("text", { x: lx(0.05) + 6, y: ly(0.86), "font-size": 11, class: "t-blue", text: "log gate τ=0.02" }, svg);
      drawIn(linear, 900, 200); drawIn(logg, 1100, 900);
    }
    function visBudget(box) {
      const runs = []; // illustrative sequence of development runs (GPU-h)
      const push = (n, kind, h) => { for (let i = 0; i < n; i++) runs.push({ kind, h }); };
      push(36, "mu1", 0.04); push(6, "mu2", 0.25); push(28, "mu1", 0.04); push(8, "mu2", 0.25); push(14, "mu1", 0.04);
      const total = 8, used = runs.reduce((a, r) => a + r.h, 0);
      box.innerHTML = `<div class="budget-meter"><div class="track"></div>
        <div class="legend"><span>used <b class="mono" id="bmUsed">0.00</b> / ${total} GPU-h</span><span><i style="display:inline-block;width:10px;height:10px;background:var(--blue);border-radius:2px"></i> μ1 <b class="mono" id="bmM1">0</b> runs</span><span><i style="display:inline-block;width:10px;height:10px;background:var(--green);border-radius:2px"></i> μ2 <b class="mono" id="bmM2">0</b> runs</span></div>
        <p class="panel-note">Illustration: development run log, with the budget charged by wall clock on the reference card. μ3 and hidden settings are used only by the judge.</p>
        <table class="mini"><thead><tr><th>Tier</th><th>Effective params</th><th>Tokens</th><th>Per run</th><th>Who can use it</th></tr></thead><tbody>
          <tr><td class="mono">μ1</td><td>about 7M</td><td>0.14B</td><td>about 2.5 min</td><td>agent self-tests, judge tuning</td></tr>
          <tr><td class="mono">μ2</td><td>about 17M</td><td>0.35B</td><td>about 15 min</td><td>agent self-tests, evaluation</td></tr>
          <tr><td class="mono">μ3</td><td>about 42M</td><td>0.84B</td><td>about 1.5 h</td><td>judge only</td></tr></tbody></table></div>`;
      const track = box.querySelector(".track");
      let acc = 0, n1 = 0, n2 = 0;
      runs.forEach((r, i) => {
        const seg = document.createElement("i");
        seg.className = r.kind;
        seg.style.left = (acc / total * 100) + "%";
        seg.style.width = `calc(${r.h / total * 100}% - 1px)`;
        acc += r.h;
        track.appendChild(seg);
        const d = 120 + i * 32;
        if (!RM) seg.animate([{ opacity: 0, transform: "scaleY(0)" }, { opacity: 1, transform: "none" }], { duration: 160, delay: d, fill: "backwards" });
        setTimeout(() => { r.kind === "mu1" ? n1++ : n2++; const u = box.querySelector("#bmUsed"); if (!u) return; box.querySelector("#bmM1").textContent = n1; box.querySelector("#bmM2").textContent = n2; u.textContent = runs.slice(0, i + 1).reduce((a, x) => a + x.h, 0).toFixed(2); }, RM ? 0 : d);
      });
      void used;
    }
    function visFiles(box) {
      const files = [["optimizer.py", "implements FunctionalRule, exports RULE", "keep"], ["hparam_space.yaml", "at most 4 tunable hyperparameters", "keep"], ["report/index.html", "design report", "keep"],
        ["references.yaml", "references: what was borrowed and where it is in the code", "keep"], ["metrics.json", "self-reported metrics: dropped; metrics are computed only by the judge", "drop"], ["notes/scratch.py", "top-level file outside the allowlist: dropped", "drop"]];
      box.innerHTML = `<div class="files">${files.map(f => `<div class="${f[2]}"><span>submission/${f[0]}</span><span>${f[1]}</span></div>`).join("")}</div>
        <p class="panel-note" style="margin-top:12px">The run record is written to agent_run.json: view manifest and hashes, prompt hash, environment-variable names, kept and dropped files, isolation level.</p>`;
      [...box.querySelectorAll(".files div")].forEach((d, i) => { fade(d, 100 + i * 140, 300, 4); if (d.classList.contains("drop")) setTimeout(() => d.classList.add("x"), RM ? 0 : 400 + i * 140); });
      fade(box.querySelector(".panel-note"), 1200, 400);
    }
    function visMeters(box) {
      const c = ex.composite;
      const rows = [["M", c.M, "SOTA margin"], ["D", c.D, "divergence"], ["Q", c.Q, "reference quality"], ["X", c.X, "local discovery"]];
      box.innerHTML = `<div class="meter-list">${rows.map(r => `<div class="meter"><span class="mono">${r[0]}</span><span class="bar"><i></i></span><span class="val">${fmt(r[1], 3)}</span></div>`).join("")}</div>
        <div class="gates" style="margin-top:14px;padding:0;border:0;background:none">${Object.entries(c.gates).map(([k, v]) => `<div class="gate ${v ? "" : "fail"}">${checkIcon(v)}<span class="gname">${k}</span><span class="gdetail">${v ? "pass" : "fail"}</span></div>`).join("")}</div>`;
      requestAnimationFrame(() => [...box.querySelectorAll(".meter .bar i")].forEach((b, i) => setTimeout(() => { b.style.width = (rows[i][1] * 100) + "%"; }, RM ? 0 : 120 + i * 220)));
    }
    function visReport(box) {
      const c = ex.composite;
      box.innerHTML = `<div style="display:grid;gap:12px"><span class="bignum" id="repNum">0.00</span>
        <p class="mono" style="font-size:13.5px;color:var(--ink-2)">100 · ${fmt(c.M, 3)}^0.5 · ${fmt(c.D, 3)}^0.3 · ${fmt(c.Q, 3)}^0.2 · (1 + 0.25 · ${fmt(c.X, 3)})</p>
        <table class="mini"><tbody>
          <tr><td>Aggregate effective speedup</td><td class="mono">${fmt(ex.M.aggregate_speedup, 3)}× (lower bound ${fmt(ex.M.aggregate_speedup_lcb, 3)}×)</td></tr>
          <tr><td>Behavioral nearest neighbor</td><td>${esc(REF_NAME[ex.D.behavioral.nearest_reference] || ex.D.behavioral.nearest_reference)}, similarity ${fmt(1 - ex.D.behavioral.D_global, 3)}</td></tr>
          <tr><td>Local discovery</td><td>${ex.X.declarations.map(d => `${esc(d.module)}: ${esc(d.status)}`).join("; ")}</td></tr>
        </tbody></table><a class="btn small" href="#scoring">See scoring details and the two negatives</a></div>`;
      countTo(box.querySelector("#repNum"), c.score, 1400, 2);
    }
  })();

  // ------------------------------------------------------------------ OJ evaluation
  (() => {
    const CASES = ["DS-GBM-001", "B-OPT-001", "B-T7-005"].map(id => allCases.find(c => c.id === id)).filter(Boolean);
    const sw = $("#ojCaseSwitch");
    const ENV = ["ABENCH_VIEW_DIR", "ABENCH_DATA_DIR", "ABENCH_WORKSPACE", "ABENCH_CHECKPOINT_DIR", "ABENCH_SUBMISSION_DIR", "ABENCH_DEVICE", "ABENCH_DEADLINE", "PATH", "HOME (temporary)"];
    const judgeTree = c => {
      const o = c.oj || {};
      return [["case/oj.yaml", "submission allowlist, judge configuration"], ["case/eval_protocol.yaml", "evaluation settings, including hidden ones"], ["case/scoring_config.yaml", "scoring weights"], ["case/metric_registry.yaml", "metric directions and units"],
        ["case/reference_pool.yaml", "reference methods"], ["case/anti_gaming.md", "anti-gaming rules"],
        ...(o.hidden_rows ? [[`sandbox/hidden/${c.id}/test.npz.enc`, `hidden set, ${o.hidden_rows} rows, stored encrypted`, "lock"]] : [["hidden settings and data", o.judge === "pending" ? "frozen together with the judge" : "", "lock"]]),
        ["~/.abench/hidden.key", "outside the repository, 0600, given only to the judge", "lock"]];
    };
    function show(c) {
      [...sw.children].forEach(b => b.setAttribute("aria-selected", b.dataset.id === c.id));
      const o = c.oj || {};
      $("#ojAgentTree").innerHTML = (o.view || []).map(f => `<li class="f ${f.startsWith("data/") ? "data" : ""}"><span class="mono">view/${esc(f)}</span>${f.startsWith("data/") && o.public_rows ? `<span class="tag green">public, ${o.public_rows} rows</span>` : ""}</li>`).join("")
        + `<li class="f w"><span class="mono">workspace/</span><span class="tag plain">writable, deleted after the run</span></li><li class="f w"><span class="mono">workspace/submission/</span><span class="tag plain">submission directory</span></li>`;
      $("#ojEnv").innerHTML = `<span class="k">environment variables (allowlist)</span>${ENV.map(e => `<span class="tag plain mono">${e}</span>`).join("")}`;
      $("#ojJudgeTree").innerHTML = judgeTree(c).map(([f, d, k]) => `<li class="f ${k || ""}"><span class="mono">${esc(f)}</span><span class="d">${esc(d)}</span></li>`).join("");
      [...document.querySelectorAll("#ojAgentTree li, #ojJudgeTree li")].forEach((n, i) => fade(n, i * 40, 240, 3));
    }
    sw.innerHTML = CASES.map(c => `<button type="button" role="tab" data-id="${c.id}">${c.id}</button>`).join("");
    sw.addEventListener("click", e => { const b = e.target.closest("button[data-id]"); if (b) show(CASES.find(c => c.id === b.dataset.id)); });
    show(CASES[0]);

    // submission kinds
    const cnt = k => allCases.filter(c => c.oj?.kind === k).length;
    $("#ojKinds").innerHTML = `<thead><tr><th>Submission</th><th>What the judge reads</th><th>Cases</th></tr></thead><tbody>${["code_plus_checkpoint", "code_only", "report_only"].map(k =>
      `<tr><td><span class="tag">${KIND_ZH[k]}</span></td><td>${KIND_READS[k]}</td><td class="mono">${cnt(k)}</td></tr>`).join("")}</tbody>`;
    const active = allCases.filter(c => c.oj?.judge && c.oj.judge !== "pending");
    const waiting = active.filter(c => c.oj?.activation === "awaiting_data").length;
    $("#ojProv").textContent = `Cases with a judge wired in: ${active.map(c => c.id).join(", ") || "none"} (${waiting} of them awaiting operator-frozen data); the other ${allCases.length - active.length} only get format checks until their judges are frozen. The metrics of "normal submission" below come from the deterministic reference agent of DS-GBM-001 (logistic regression on synthetic micro data) and only validate the pipeline.`;

    // run-path animation
    const svg = $("#ojPipe");
    const ST = [["Build view", "allowlisted copy, read-only"], ["Run agent", "view + workspace"], ["Collect", "allowlist, drop self-reports"], ["Format check", "layout, report, references"], ["Decrypt hidden set", "judge holds the key"], ["Prediction subprocess", "gets only submission and features"], ["Compute metrics", "labels only in the parent"], ["evaluation.json", "judge_real"]];
    const X = i => 70 + i * 140, Y = 118;
    S("rect", { x: 0, y: 30, width: X(1) + 70, height: 176, rx: 10, css: "fill:var(--blue-soft);opacity:.55" }, svg);
    S("text", { x: 14, y: 50, "font-size": 12, class: "t-mono", text: "inside the sandbox" }, svg);
    S("text", { x: X(2) - 56, y: 50, "font-size": 12, class: "t-mono", text: "outside the sandbox · after the agent ends and the workspace is deleted" }, svg);
    S("line", { x1: X(1) + 70, x2: X(1) + 70, y1: 22, y2: 214, css: "stroke:var(--ink)", "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
    const track = S("line", { x1: X(0), x2: X(7), y1: Y, y2: Y, class: "axis" }, svg);
    const nodes = ST.map(([a, b], i) => {
      const g = S("g", {}, svg);
      const c = S("circle", { cx: X(i), cy: Y, r: 15, css: "fill:var(--sheet);stroke:var(--ink-3)", "stroke-width": 2 }, g);
      S("text", { x: X(i), y: Y + 4.5, "text-anchor": "middle", "font-size": 12, class: "t-mono t-ink", text: i + 1 }, g);
      S("text", { x: X(i), y: Y - 30, "text-anchor": "middle", "font-size": 13.5, class: "t-ink", text: a }, g);
      S("text", { x: X(i), y: Y + 40, "text-anchor": "middle", "font-size": 11.5, text: b }, g);
      const note = S("text", { x: X(i), y: Y + 62, "text-anchor": "middle", "font-size": 12, text: "" }, g);
      return { c, note };
    });
    const dot = S("circle", { cx: X(0), cy: Y, r: 7, css: "fill:var(--blue)" }, svg);
    const SC = [
      { id: "ok", name: "Normal submission", stop: null, notes: { 2: ["kept 4 files", "green"] }, verdict: "evaluated", vtext: "evaluated", metrics: { accuracy: 0.96875, roc_auc: 0.994135, log_loss: 0.173018 } },
      { id: "selfrep", name: "Smuggled metrics.json", stop: null, notes: { 2: ["dropped metrics.json", "amber"] }, verdict: "evaluated", vtext: "the self-reported file is dropped; metrics are still computed by the judge", metrics: { accuracy: 0.96875, roc_auc: 0.994135, log_loss: 0.173018 } },
      { id: "exp", name: "Report with an experiments section", stop: 3, notes: { 3: ["section '4 Experimental Results'", "red"] }, verdict: "submission_invalid", vtext: "the report covers design rationale only, not experiments" },
      { id: "smoke", name: "Smoke run", stop: 4, notes: { 4: ["no decryption", "amber"] }, verdict: "smoke_unscored", vtext: "validates process and format only; unscored" },
      { id: "nowrap", name: "Benchmark without isolation", stop: 0, notes: { 0: ["refused to start", "red"] }, verdict: "refused", vtext: "benchmark requires --agent-wrapper" },
    ];
    const ssw = $("#ojScenario");
    ssw.innerHTML = SC.map(s => `<button type="button" role="tab" data-id="${s.id}">${s.name}</button>`).join("");
    let run = 0;
    async function play(s) {
      const my = ++run;
      [...ssw.children].forEach(b => b.setAttribute("aria-selected", b.dataset.id === s.id));
      nodes.forEach(n => { n.c.style.fill = "var(--sheet)"; n.c.style.stroke = "var(--ink-3)"; n.note.textContent = ""; n.note.removeAttribute("class"); });
      $("#ojVerdict").innerHTML = "";
      dot.setAttribute("cx", X(0)); dot.style.fill = "var(--blue)";
      const last = s.stop == null ? 7 : s.stop;
      for (let i = 0; i <= last; i++) {
        if (my !== run) return;
        if (i > 0) await tween(420, t => dot.setAttribute("cx", X(i - 1) + (X(i) - X(i - 1)) * t));
        if (my !== run) return;
        const n = s.notes[i];
        const fail = s.stop === i;
        nodes[i].c.style.fill = fail ? "var(--red-soft)" : "var(--green-soft)";
        nodes[i].c.style.stroke = fail ? "var(--red)" : "var(--green)";
        if (n) { nodes[i].note.textContent = n[0]; nodes[i].note.setAttribute("class", "t-" + (n[1] === "green" ? "green" : n[1] === "amber" ? "amber" : "red")); }
        if (fail) dot.style.fill = "var(--red)";
        await sleep(fail ? 0 : 120);
      }
      if (my !== run) return;
      const ok = s.verdict === "evaluated";
      $("#ojVerdict").innerHTML = `<span class="stamp ${ok ? "ok" : ""}">${s.verdict}</span><span>${esc(s.vtext)}</span>${s.metrics ? `<span class="mets">${Object.entries(s.metrics).map(([k, v]) => `<span><span class="k">${k}</span><b class="mono">${v.toFixed(3)}</b></span>`).join("")}</span>` : ""}`;
      fade($("#ojVerdict"), 0, 300);
    }
    ssw.addEventListener("click", e => { const b = e.target.closest("button[data-id]"); if (b) play(SC.find(s => s.id === b.dataset.id)); });
    onView(svg, () => play(SC[0]), 0.4);
    if (RM) play(SC[0]);
  })();

  // ------------------------------------------------------------------ design report preview
  (() => {
    const DOCS = [["reports/B-OPT-001-rankgate.html", "RankGate example", "cases/B-OPT-001/example_submission/report/index.html"], ["reports/design_template.html", "Design report template", "templates/report_template.html"], ["reports/judgment_template.html", "Judgment report template", "templates/report_template.html (T7)"]]
      .filter(d => (A.reports || []).includes(d[0]));
    const sw = $("#repSwitch"), fr = $("#repFrame");
    const ANCH = ["abstract1", "S1", "S2", "S3", "S4", "bib"];
    const CLS = ["ltx_document", "ltx_title_document", "ltx_abstract", "ltx_section", "ltx_equation", "ltx_float_algorithm", "ltx_theorem", "ltx_bibliography"];
    $("#repAnchors").innerHTML = ANCH.map(a => `<span class="tag plain mono" data-a="${a}">#${a}</span>`).join("");
    $("#repClasses").innerHTML = CLS.map(a => `<span class="tag plain mono" data-c="${a}">.${a}</span>`).join("");
    if (!DOCS.length) { fr.closest(".rep-frame").innerHTML = `<p class="panel-note">No report copies found; run python3 tools/build_site_data.py.</p>`; return; }
    sw.innerHTML = DOCS.map((d, i) => `<button type="button" role="tab" data-i="${i}" aria-selected="${i === 0}">${d[1]}</button>`).join("");
    sw.addEventListener("click", e => {
      const b = e.target.closest("button[data-i]"); if (!b) return;
      const d = DOCS[+b.dataset.i];
      [...sw.children].forEach(x => x.setAttribute("aria-selected", x === b));
      fr.src = d[0]; $("#repOpen").href = d[0]; $("#repUrl").textContent = d[2];
    });
    fr.addEventListener("load", () => {
      let doc = null;
      try { doc = fr.contentDocument; } catch (e) { doc = null; }
      const live = $("#repLive");
      document.querySelectorAll("#repAnchors .tag, #repClasses .tag").forEach(t => { t.className = "tag plain mono"; });
      if (!doc || !doc.body) { live.textContent = "The browser does not allow reading the preview content (for example when index.html is opened by double-clicking); the structure check is done by validate_submission at submission time."; return; }
      const mark = (sel, ok, i) => setTimeout(() => { sel.className = "tag mono " + (ok ? "green" : "red"); }, RM ? 0 : i * 110);
      [...document.querySelectorAll("#repAnchors .tag")].forEach((t, i) => mark(t, !!doc.getElementById(t.dataset.a), i));
      [...document.querySelectorAll("#repClasses .tag")].forEach((t, i) => mark(t, !!doc.querySelector("." + t.dataset.c), i + ANCH.length));
      const heads = [...doc.querySelectorAll("h1,h2,h3,h4,h5,h6")].map(h => h.textContent);
      const bad = heads.filter(h => /experiment|result|ablation|evaluation|benchmark|metric|leaderboard/i.test(h));
      const nb = doc.querySelectorAll("li.ltx_bibitem").length, ne = doc.querySelectorAll("table.ltx_equation").length;
      live.textContent = `Read live from the preview: ${heads.length} headings, ${ne} numbered equations, ${nb} references; ${bad.length} forbidden sections.`;
    });
  })();

  // ------------------------------------------------------------------ evaluation modes
  (() => {
    const subs = C.submissions;
    const order = ["example_submission", "clone_muon", "rankgate_linear_gate"];
    let curSub = "example_submission", curSet = "L2_main";
    const sw = $("#subSwitch");
    sw.innerHTML = order.map(k => {
      const c = subs[k].composite;
      const failed = Object.entries(c.gates).filter(([, v]) => !v).map(([g]) => g);
      return `<button class="sub-btn" role="tab" type="button" data-k="${k}" aria-selected="${k === curSub}"><span class="s1">${esc(subs[k].label)}</span><span class="s2">${failed.length ? "0 points · failed: " + failed.join(", ") : "overall " + fmt(c.score, 2)}</span></button>`;
    }).join("");
    sw.addEventListener("click", e => { const b = e.target.closest("[data-k]"); if (!b) return; curSub = b.dataset.k; [...sw.children].forEach(x => x.setAttribute("aria-selected", x === b)); renderAll(); });
    const ss = $("#settingSwitch");
    ss.innerHTML = Object.keys(SETTING_NAME).map(k => `<button type="button" role="tab" data-s="${k}" aria-selected="${k === curSet}">${SETTING_NAME[k]}</button>`).join("");
    ss.addEventListener("click", e => { const b = e.target.closest("[data-s]"); if (!b) return; curSet = b.dataset.s; [...ss.children].forEach(x => x.setAttribute("aria-selected", x === b)); renderEnv(); });

    const GATE_ZH = { "aggregate_speedup": "aggregate speedup", lcb: "lower confidence bound", per_setting: "single setting", ood_regression: "OOD degradation", "step_time": "step time" };
    function renderFormula() {
      const s = subs[curSub], c = s.composite;
      const pass = Object.values(c.gates).every(Boolean);
      const failed = Object.entries(c.gates).filter(([, v]) => !v).map(([g]) => g);
      const e = c.exponents;
      $("#formula").innerHTML = `<div class="score-row"><span class="score ${pass ? "" : "zero"}" id="scoreNum">0.00</span>
        <span class="why">${pass ? "all three thresholds pass" : `thresholds failed: ${failed.join(", ")}. If they passed, the score would be ${fmt(c.score_if_gates_pass, 2)}`}</span></div>
        <div class="expr"><span>100</span><span>·</span>
          <span class="term ${pass ? "" : "off"}">M<sup>${e.M}</sup> = <b>${fmt(c.M, 3)}</b></span><span>·</span>
          <span class="term ${pass ? "" : "off"}">D<sup>${e.D}</sup> = <b>${fmt(c.D, 3)}</b></span><span>·</span>
          <span class="term ${pass ? "" : "off"}">Q<sup>${e.Q}</sup> = <b>${fmt(c.Q, 3)}</b></span><span>·</span>
          <span class="term ${pass ? "" : "off"}">(1 + ${c.discovery_bonus}·X), X = <b>${fmt(c.X, 3)}</b></span></div>`;
      countTo($("#scoreNum"), c.score, 900, 2);
      // threshold details
      const M = s.M, D = s.D;
      const sotaFail = Object.entries(M.gate_checks).filter(([, v]) => !v).map(([k]) => GATE_ZH[k.split(/[<>=]/)[0]] || k);
      const rows = [
        ["L0", c.gates.L0, `interface, memory (state ${s.L0.state_ratio_max}× parameters), numerics, determinism`],
        ["SOTA", c.gates.SOTA, `aggregate effective speedup ${fmt(M.aggregate_speedup, 3)}×, lower bound ${fmt(M.aggregate_speedup_lcb, 3)}×` + (sotaFail.length ? `; not met: ${sotaFail.join(", ")}` : "; every setting ≥ 0.9×, OOD degradation ≤ 1%")],
        ["differentiated", c.gates["differentiated"], `behavioral divergence ${fmt(D.behavioral.D_global, 3)}, clone threshold ${fmt(D.threshold, 3)} (hyperparameter-perturbation noise floor ${fmt(D.noise_floor, 3)} × 1.5); D = ${fmt(D.D, 3)} ≥ 0.2`]
      ];
      $("#gates").innerHTML = rows.map(([n, ok, d], i) => `<div class="gate ${ok ? "" : "fail"}" data-i="${i}">${checkIcon(ok)}<span class="gname">${n}</span><span class="gdetail">${esc(d)}</span></div>`).join("");
      [...$("#gates").children].forEach((g, i) => fade(g, 100 + i * 140, 300, 0));
    }

    function renderEnv() {
      const svg = $("#envChart"); svg.innerHTML = "";
      const set = C.curves[curSet];
      const W = 960, H = 380, L = 70, R = 90, T = 40, B = 48;
      const refs = Object.entries(set.references);
      const pts = refs.map(([k, v]) => ({ k, xs: v.budgets.map(b => Math.log(b * set.time_ratios[k])), ys: v.loss }));
      const sub = set.submissions[curSub];
      const subMean = sub.loss.reduce((a, b) => a + b, 0) / sub.loss.length;
      const allY = pts.flatMap(p => p.ys).concat(sub.loss);
      const yMin = Math.min(...allY), yMax = Math.max(...allY), pad = (yMax - yMin) * 0.08;
      const xMin = Math.log(0.55), xMax = Math.log(2.1);
      const X = v => L + (v - xMin) / (xMax - xMin) * (W - L - R);
      const Y = v => T + (yMax + pad - v) / (yMax - yMin + 2 * pad) * (H - T - B);
      // axes and grid
      [0.5, 0.7, 1, 1.4, 2].forEach(b => {
        S("line", { x1: X(Math.log(b)), x2: X(Math.log(b)), y1: T, y2: H - B, class: "grid" }, svg);
        S("text", { x: X(Math.log(b)), y: H - B + 18, "text-anchor": "middle", "font-size": 11.5, class: "t-mono", text: b + "×" }, svg);
      });
      S("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, text: "cost = token-budget multiple × step-time ratio (log axis)" }, svg);
      const nT = 5;
      for (let i = 0; i <= nT; i++) {
        const v = yMin - pad + i / nT * (yMax - yMin + 2 * pad);
        S("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "grid" }, svg);
        S("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono", text: v.toFixed(3) }, svg);
      }
      S("text", { x: L, y: T - 16, "font-size": 12, text: "final validation loss (after learning-rate decay)" }, svg);
      const clipId = "envClip";
      const cp = S("clipPath", { id: clipId }, S("defs", {}, svg));
      S("rect", { x: L, y: T, width: W - L - R, height: H - T - B }, cp);
      // reference curves
      const refPaths = pts.map(p => {
        const d = p.xs.map((x, i) => (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(p.ys[i]).toFixed(1)).join("");
        const path = S("path", { d, fill: "none", css: "stroke:var(--ink-3)", "stroke-width": 1, opacity: 0.6, "clip-path": `url(#${clipId})` }, svg);
        S("title", { text: REF_NAME[p.k] || p.k }, path);
        const last = p.xs.length - 1;
        if (["muon", "adamw"].includes(p.k)) S("text", { x: Math.min(X(p.xs[last]), W - R) + 6, y: Y(p.ys[last]) + (p.k === "muon" ? 12 : -2), "font-size": 11.5, text: REF_NAME[p.k] }, svg);
        return path;
      });
      // envelope
      const N = 220, gx = [], env = [];
      const gx0 = Math.max(xMin, ...pts.map(p => p.xs[0])), gx1 = Math.min(xMax, ...pts.map(p => p.xs.at(-1)));
      for (let i = 0; i <= N; i++) {
        const x = gx0 + i / N * (gx1 - gx0);
        let best = Infinity;
        pts.forEach(p => {
          if (x < p.xs[0] || x > p.xs.at(-1)) return;
          let j = 1; while (j < p.xs.length - 1 && p.xs[j] < x) j++;
          const t = (x - p.xs[j - 1]) / (p.xs[j] - p.xs[j - 1]);
          best = Math.min(best, p.ys[j - 1] + t * (p.ys[j] - p.ys[j - 1]));
        });
        gx.push(x); env.push(best);
      }
      const envD = gx.map((x, i) => (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(env[i]).toFixed(1)).join("");
      const envPath = S("path", { d: envD, fill: "none", css: "stroke:var(--blue)", "stroke-width": 3, "stroke-linejoin": "round", "clip-path": `url(#${clipId})` }, svg);
      S("text", { x: X(gx[Math.round(N * 0.18)]), y: Y(env[Math.round(N * 0.18)]) - 12, "font-size": 12, class: "t-blue", text: "reference envelope" }, svg);
      S("title", { text: "reference envelope: the lowest loss over all reference methods at each cost point" }, envPath);
      // submission point and reading
      const sx = X(Math.log(sub.time_ratio)), sy = Y(subMean);
      let cross = null;
      const runMin = []; env.reduce((m, v, i) => (runMin[i] = Math.min(m, v)), Infinity);
      for (let i = 0; i < runMin.length; i++) if (runMin[i] <= subMean) { cross = i === 0 ? gx[0] : gx[i - 1] + (runMin[i - 1] - subMean) / (runMin[i - 1] - runMin[i]) * (gx[i] - gx[i - 1]); break; }
      const g = S("g", {}, svg);
      const report = C.submissions[curSub].M.per_setting[curSet];
      let readTxt;
      if (cross != null) {
        const cx = X(cross);
        S("path", { d: `M${sx},${sy} L${cx},${sy} L${cx},${H - B}`, fill: "none", css: "stroke:var(--red)", "stroke-width": 1.4, "stroke-dasharray": "4 3" }, g);
        S("circle", { cx, cy: sy, r: 3.5, css: "fill:var(--red)" }, g);
        S("text", { x: cx + 6, y: H - B - 8, "font-size": 11.5, class: "t-red", text: "envelope needs " + Math.exp(cross).toFixed(2) + "× cost" }, g);
        readTxt = `The submission reaches ${subMean.toFixed(4)} at 1× token budget with step time ${fmt(sub.time_ratio, 2)}×; the reference envelope needs ${Math.exp(cross).toFixed(2)}× the cost to reach it, so the effective speedup is ${fmt(report.speedup, 3)}× (reported value, geometric mean over seeds).`;
      } else {
        S("path", { d: `M${sx},${sy} L${W - R},${sy}`, fill: "none", css: "stroke:var(--red)", "stroke-width": 1.4, "stroke-dasharray": "4 3" }, g);
        readTxt = `The submission's loss is below the lowest value the reference envelope reaches within 2× cost, so the speedup is counted at the cap of 2.`;
      }
      sub.loss.forEach(l => S("circle", { cx: sx, cy: Y(l), r: 2.6, css: "fill:var(--ink);opacity:.45" }, g));
      const dot = S("circle", { cx: sx, cy: sy, r: 6.5, css: "fill:var(--ink);stroke:var(--sheet)", "stroke-width": 2 }, g);
      S("text", { x: sx - 10, y: sy - 12, "text-anchor": "end", "font-size": 12, class: "t-ink", "font-weight": 700, text: subs[curSub].label.replace(/\s*\(.*\)/, "") }, g);
      // animation
      refPaths.forEach((p, i) => drawIn(p, 600, i * 40));
      drawIn(envPath, 1000, 300);
      if (!RM) g.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 1100, fill: "backwards" });
      pop(dot, 1100, 420);
      $("#envNote").textContent = `${SETTING_NAME[curSet]}: ${SETTING_DESC[curSet]}. ${readTxt}`;
    }

    function renderDiv() {
      const D = subs[curSub].D, b = D.behavioral;
      const thrSim = 1 - D.threshold;
      const rows = Object.entries(b.similarity_by_reference).sort((a, c) => c[1] - a[1]);
      $("#divBars").innerHTML = rows.map(([k, v]) => `<div class="brow ${v > thrSim ? "hot" : ""}"><span class="bname">${esc(REF_NAME[k] || k)}</span><span class="btrack"><i data-w="${(v * 100).toFixed(1)}"></i><span class="thr" style="left:${(thrSim * 100).toFixed(1)}%" title="clone threshold"></span></span><span class="bval">${fmt(v, 3)}</span></div>`).join("");
      requestAnimationFrame(() => $("#divBars").querySelectorAll("i").forEach((i, n) => setTimeout(() => { i.style.width = i.dataset.w + "%"; }, RM ? 0 : n * 60)));
      $("#divNote").innerHTML = `The red dashed line is the clone threshold: similarity above it means the difference from that reference method is no larger than "changing a hyperparameter". Behavioral divergence ${fmt(b.D_global, 3)}, layer-wise splice view ${fmt(b.D_patch, 3)}; code divergence ${fmt(D.code, 3)}; mechanism divergence ${fmt(D.mech, 3)} (judge levels ${D.mech_levels.join(", ")}). D = ${fmt(D.D, 3)}.`;
    }
    function renderQ() {
      const Q = subs[curSub].Q;
      const rows = Q.sources;
      $("#qBars").innerHTML = rows.map(s => {
        const p = resolveSrc(s.id);
        const name = p ? (PAPER_SHORT[p.id] || p.id) : s.id.replace(/^arXiv:/, "");
        const cls = s.status !== "ok" ? "hot" : s.usage === "decorative" ? "muted" : s.usage === "core" ? "" : "good";
        const usage = { core: "core", partial: "partial", decorative: "decorative" }[s.usage] || s.usage;
        if (s.status !== "ok") return `<div class="brow hot" title="${esc(s.id)}"><span class="bname">${esc(name)} · unresolvable</span><span class="btrack"><i data-w="100"></i></span><span class="bval">×0.8</span></div>`;
        const right = s.status === "unresolvable" ? "×0.8" : fmt(s.contribution, 2);
        return `<div class="brow ${cls}" title="${esc(s.title || s.id)}"><span class="bname">${esc(name)} · ${usage}</span><span class="btrack"><i data-w="${(s.status === "ok" ? (s.usage === "decorative" ? s.w * 100 : s.contribution * 100) : 100).toFixed(1)}"></i></span><span class="bval">${right}</span></div>`;
      }).join("");
      requestAnimationFrame(() => $("#qBars").querySelectorAll("i").forEach((i, n) => setTimeout(() => { i.style.width = i.dataset.w + "%"; }, RM ? 0 : n * 60)));
      const hidden = Q.unresolved_source_count || 0;
      const hiddenNote = hidden ? `The historical scoring also has ${hidden} unresolvable source(s), hidden from the chart; the aggregate Q keeps the original penalty.` : "";
      $("#qNote").innerHTML = `Bar length = paper weight w × usage. Gray bars are decorative citations (only w shown, contribution 0); red bars are unresolvable sources (overall × 0.8). Q = 1 − Π(1 − w·u) × penalty = ${fmt(Q.Q, 3)}. ${hiddenNote}`;
    }
    function renderX() {
      const X = subs[curSub].X;
      const box = $("#xCards");
      if (!X.declarations.length) { box.innerHTML = `<p class="empty-note">This submission claims no local discovery: X = 0, with no effect on the thresholds.</p>`; return; }
      const tagCls = st => st.startsWith("local discovery") ? "green" : st.startsWith("transfer") ? "" : st.startsWith("retrieval reuse") ? "amber" : "red";
      box.innerHTML = X.declarations.map(d => `<div class="x-card"><div class="xh"><span class="xm">${esc(d.module)}</span><span class="tag ${tagCls(d.status)}">${esc(d.status)}</span></div>
        <p>Source ${esc(d.source)} (${esc(d.source_year)}, ${esc(d.source_field)}): ${esc(d.transferred || "")}</p>
        <div class="xf"><span>full ${fmt(d.S_full, 3)}×</span><span>ablated ${fmt(d.S_ablated, 3)}×</span><span>effect ${fmt(d.effect_ln, 3)}</span><span>a ${fmt(d.a, 2)}</span><span>n ${d.n}</span><span>c ${d.c}</span><span>r ${d.r}</span><span>x ${fmt(d.x, 3)}</span></div>
        ${d.ml_precedent ? `<p>Judge search: ${esc(d.ml_precedent)}</p>` : ""}</div>`).join("") + `<p class="empty-note" style="grid-column:1/-1">X = ${fmt(X.X, 3)}, overall score multiplied by ${(1 + 0.25 * X.X).toFixed(3)}.</p>`;
      [...box.querySelectorAll(".x-card")].forEach((c, i) => fade(c, i * 120, 350));
    }
    function renderAll() { renderFormula(); renderEnv(); renderDiv(); renderQ(); renderX(); }
    let started = false;
    onView($("#scoring"), () => { started = true; renderAll(); }, 0.15);
    // render the static result first when out of view, to avoid a blank
    if (RM) renderAll();
    void started;
  })();

  // ------------------------------------------------------------------ Sinkhorn-Knopp
  (() => {
    const n = 5;
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const M0 = Array.from({ length: n }, () => Array.from({ length: n }, () => 0.15 + rnd() * 2.6));
    let M = M0.map(r => r.slice()), iter = 0, half = 0, busy = false, playing = false;
    const svg = $("#skSvg");
    const cs = 56, ox = 40, oy = 34;
    const rowSum = m => m.map(r => r.reduce((a, b) => a + b, 0));
    const colSum = m => m[0].map((_, j) => m.reduce((a, r) => a + r[j], 0));
    const err = m => rowSum(m).reduce((a, v) => a + Math.abs(v - 1), 0) + colSum(m).reduce((a, v) => a + Math.abs(v - 1), 0);
    let cells = [], rTxt = [], cTxt = [], sweep;
    function build() {
      svg.innerHTML = "";
      S("text", { x: ox + n * cs + 30, y: oy - 12, "text-anchor": "middle", "font-size": 11.5, text: "row sum" }, svg);
      S("text", { x: ox - 10, y: oy + n * cs + 30, "text-anchor": "end", "font-size": 11.5, text: "col sum" }, svg);
      sweep = S("rect", { x: ox, y: oy, width: 0, height: 0, rx: 4, css: "fill:none;stroke:var(--red)", "stroke-width": 2, opacity: 0 }, svg);
      cells = [];
      for (let i = 0; i < n; i++) {
        cells.push([]);
        for (let j = 0; j < n; j++) {
          const r = S("rect", { x: ox + j * cs + 2, y: oy + i * cs + 2, width: cs - 4, height: cs - 4, rx: 4, css: "fill:var(--blue)" }, svg);
          const t = S("text", { x: ox + j * cs + cs / 2, y: oy + i * cs + cs / 2 + 4, "text-anchor": "middle", "font-size": 12, class: "t-mono" }, svg);
          cells[i].push([r, t]);
        }
      }
      rTxt = Array.from({ length: n }, (_, i) => S("text", { x: ox + n * cs + 30, y: oy + i * cs + cs / 2 + 4, "text-anchor": "middle", "font-size": 12.5, class: "t-mono" }, svg));
      cTxt = Array.from({ length: n }, (_, j) => S("text", { x: ox + j * cs + cs / 2, y: oy + n * cs + 30, "text-anchor": "middle", "font-size": 12.5, class: "t-mono" }, svg));
      svg.appendChild(sweep);
    }
    function paint(m) {
      const mx = Math.max(...m.flat());
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const [r, t] = cells[i][j];
        r.style.opacity = (0.12 + 0.88 * m[i][j] / mx).toFixed(3);
        t.textContent = m[i][j].toFixed(2);
        t.setAttribute("class", "t-mono " + (m[i][j] / mx > 0.55 ? "" : "t-ink"));
        t.style.fill = m[i][j] / mx > 0.55 ? "var(--sheet)" : "";
      }
      rowSum(m).forEach((v, i) => { rTxt[i].textContent = v.toFixed(2); rTxt[i].setAttribute("class", "t-mono " + (Math.abs(v - 1) < 0.01 ? "t-green" : "t-red")); });
      colSum(m).forEach((v, j) => { cTxt[j].textContent = v.toFixed(2); cTxt[j].setAttribute("class", "t-mono " + (Math.abs(v - 1) < 0.01 ? "t-green" : "t-red")); });
      $("#skErr").textContent = err(m).toFixed(4);
    }
    async function stepOnce() {
      if (busy) return; busy = true;
      const rowsStep = half % 2 === 0;
      const target = M.map(r => r.slice());
      if (rowsStep) { const rs = rowSum(M); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) target[i][j] = M[i][j] / rs[i]; }
      else { const c = colSum(M); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) target[i][j] = M[i][j] / c[j]; }
      $("#skOp").textContent = rowsStep ? "row normalization: divide each row by its sum" : "column normalization: divide each column by its sum";
      if (rowsStep) iter++;
      $("#skIter").textContent = iter;
      const from = M.map(r => r.slice());
      if (!RM) {
        sweep.setAttribute("opacity", 1);
        await tween(380, t => {
          if (rowsStep) { sweep.setAttribute("x", ox); sweep.setAttribute("width", n * cs); sweep.setAttribute("height", cs); sweep.setAttribute("y", oy + Math.min(n - 1, Math.floor(t * n)) * cs); }
          else { sweep.setAttribute("y", oy); sweep.setAttribute("height", n * cs); sweep.setAttribute("width", cs); sweep.setAttribute("x", ox + Math.min(n - 1, Math.floor(t * n)) * cs); }
        }, t => t);
      }
      await tween(520, t => { const cur = from.map((r, i) => r.map((v, j) => v + (target[i][j] - v) * t)); paint(cur); });
      sweep.setAttribute("opacity", 0);
      M = target; half++; busy = false;
    }
    function reset() { playing = false; $("#skPlay").textContent = "Play"; M = M0.map(r => r.slice()); iter = 0; half = 0; $("#skIter").textContent = 0; $("#skOp").textContent = "original matrix: positive, arbitrary row and column sums"; paint(M); }
    async function play() {
      if (playing) { playing = false; $("#skPlay").textContent = "Play"; return; }
      playing = true; $("#skPlay").textContent = "Pause";
      if (err(M) < 1e-3) reset(), playing = true, $("#skPlay").textContent = "Pause";
      while (playing && err(M) > 1e-3 && half < 24) { await stepOnce(); await sleep(260); }
      playing = false; $("#skPlay").textContent = "Play";
      if (err(M) <= 1e-3) $("#skOp").textContent = "converged: all row and column sums are 1, giving a doubly stochastic matrix";
    }
    build(); reset();
    $("#skPlay").addEventListener("click", play);
    $("#skStep").addEventListener("click", () => { playing = false; $("#skPlay").textContent = "Play"; stepOnce(); });
    $("#skReset").addEventListener("click", reset);
    onView($("#skSvg"), () => { if (!RM) play(); }, 0.4);
  })();

  // ------------------------------------------------------------------ verification modes
  (() => {
    // Budget ladder
    const lad = $("#ladderSvg");
    const tiers = [["μ1", 7, "about 7M", "2.5 min", false], ["μ2", 17, "about 17M", "15 min", false], ["μ3", 42, "about 42M", "1.5 h", false], ["review", 135, "135M", "unscored", true]];
    const hOf = p => 30 + (Math.log10(p) - Math.log10(5)) / (Math.log10(150) - Math.log10(5)) * 120;
    const bars = tiers.map(([name, p, lab, t, ghost], i) => {
      const x = 22 + i * 84, h = hOf(p), y = 170 - h;
      const g = S("g", {}, lad);
      S("rect", { x, y, width: 58, height: h, rx: 3, css: ghost ? "fill:none;stroke:var(--ink-3)" : `fill:var(--blue);opacity:${0.45 + i * 0.18}`, "stroke-dasharray": ghost ? "4 3" : null, "stroke-width": 1.4 }, g);
      S("text", { x: x + 29, y: y - 8, "text-anchor": "middle", "font-size": 12, class: "t-mono t-ink", text: lab }, g);
      S("text", { x: x + 29, y: 190, "text-anchor": "middle", "font-size": 12.5, class: "t-ink", text: name }, g);
      S("text", { x: x + 29, y: 207, "text-anchor": "middle", "font-size": 11, text: t }, g);
      return g;
    });
    S("line", { x1: 12, x2: 348, y1: 170, y2: 170, class: "axis" }, lad);
    onView(lad, () => bars.forEach((g, i) => { if (!RM) { g.style.transformBox = "fill-box"; g.style.transformOrigin = "bottom"; g.animate([{ transform: "scaleY(0)", opacity: 0 }, { transform: "none", opacity: 1 }], { duration: 600, delay: i * 220, easing: "cubic-bezier(.2,.8,.2,1)", fill: "backwards" }); } }));

    // Pro timeline
    const pro = $("#proSvg");
    const x0 = 34, x1 = 340, xOf = h => x0 + h / 24 * (x1 - x0);
    for (let k = 0; k < 4; k++) S("text", { x: 4, y: 40 + k * 22 + 4, "font-size": 10.5, class: "t-mono", text: "GPU" + k }, pro);
    const segs = [[0.2, 3.0, 0.55], [3.3, 6.5, 0.7], [7, 9.4, 0.6], [10, 23.4, 0.92]];
    const lanes = [];
    for (let k = 0; k < 4; k++) segs.forEach(([a, b, u]) => lanes.push(S("rect", { x: xOf(a), y: 32 + k * 22, width: xOf(b) - xOf(a), height: 14, rx: 2, css: `fill:var(--blue);opacity:${u}` }, pro)));
    S("line", { x1: x0, x2: x1, y1: 140, y2: 140, class: "axis" }, pro);
    [0, 6, 12, 18, 24].forEach(h => S("text", { x: xOf(h), y: 212, "text-anchor": "middle", "font-size": 10.5, class: "t-mono", text: h + "h" }, pro));
    const ticks = [];
    for (let i = 1; i <= 48; i++) ticks.push(S("rect", { x: xOf(i / 2) - 2, y: 136, width: 4, height: 8, css: "fill:var(--ink-2)" }, pro));
    S("text", { x: x0, y: 186, "font-size": 11, text: "checkpoint hash registered every 30 minutes (48 in total)" }, pro);
    const pick = 35; // hour 17.5
    const br = S("path", { d: `M${xOf(pick / 2)},${150} L${xOf(pick / 2)},${156} L${xOf(pick / 2 + 1)},${156} L${xOf(pick / 2 + 1)},${150}`, fill: "none", css: "stroke:var(--red)", "stroke-width": 2 }, pro);
    const brT = S("text", { x: xOf(pick / 2 + 0.5), y: 170, "text-anchor": "middle", "font-size": 11, class: "t-red", text: "sampled 1 h resume check" }, pro);
    const cursor = S("line", { x1: x0, x2: x0, y1: 26, y2: 146, css: "stroke:var(--ink)", "stroke-width": 1.2 }, pro);
    onView(pro, async () => {
      if (RM) return;
      lanes.forEach(r => r.style.opacity = 0); ticks.forEach(t => t.style.opacity = 0); br.style.opacity = 0; brT.style.opacity = 0;
      await tween(2600, t => {
        const h = t * 24, x = xOf(h);
        cursor.setAttribute("x1", x); cursor.setAttribute("x2", x);
        lanes.forEach(r => { const rx = +r.getAttribute("x"); r.style.opacity = rx < x ? "" : 0; });
        ticks.forEach((tk, i) => { tk.style.opacity = xOf((i + 1) / 2) <= x ? 1 : 0; });
      }, t => t);
      ticks[pick - 1].style.fill = "var(--red)";
      br.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, fill: "forwards" });
      brT.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: 150, fill: "forwards" });
    }, 0.4);

    // judgment
    const claims = [
      { c: "Matrix optimizers give a 1.4–2× token-efficiency gain over AdamW", src: "B-T7-005 · checked against Fantastic Pretraining Optimizers (2509.02046)", v: "partly holds", why: "With tuned baselines, about 1.4× appears only at the 0.1B scale and shrinks to about 1.1× at 1.2B", ok: false },
      { c: "A newly proposed 3D segmentation architecture beats nnU-Net across the board", src: "MD-T7-002 · checked against nnU-Net Revisited (2404.09556)", v: "mostly does not hold", why: "Under rigorous validation, a well-configured nnU-Net remains a very strong baseline", ok: false },
      { c: "A tuned GBDT remains a strong baseline on medium-sized tabular data", src: "DS-T7-003 · checked against Grinsztajn et al. (2207.08815)", v: "holds", why: "On the medium-sized datasets of their benchmark, tree models lead deep tabular models overall", ok: true }
    ];
    const jd = $("#judgeDemo");
    let ci = 0, jt = null;
    function showClaim() {
      const k = claims[ci % claims.length]; ci++;
      jd.innerHTML = `<div class="claim"><span class="src">${esc(k.src)}</span><span>"${esc(k.c)}"</span></div>
        <div class="verdict"><span class="stamp ${k.ok ? "ok" : ""}">${esc(k.v)}</span><span style="color:var(--ink-2);font-size:13.5px">${esc(k.why)}</span></div>
        <p class="panel-note">Scoring: whether the judgment is correct, plus the information content of the minimal verification plan. The agent submits only a judgment report and runs no experiments.</p>`;
      fade(jd.querySelector(".claim"), 0, 300);
      const st = jd.querySelector(".stamp");
      if (!RM) st.animate([{ opacity: 0, transform: "rotate(-4deg) scale(1.8)" }, { opacity: 1, transform: "rotate(-4deg) scale(1)" }], { duration: 420, delay: 500, easing: "cubic-bezier(.2,1.3,.4,1)", fill: "backwards" });
    }
    showClaim();
    onView(jd, () => { if (!RM) jt = setInterval(showClaim, 5200); });
    void jt;
  })();

  // ------------------------------------------------------------------ resource tiers, matrix
  const caps = { low: ["≤ 2", "≤ 0.5", "≤ 10", "GPU-h (reference card)"], medium: ["≤ 4", "≤ 2", "≤ 30", "GPU-h (reference card)"], high: ["≤ 8", "≤ 6", "≤ 80", "GPU-h (reference card)"], pro: ["96 (4×H100×24h)", "≤ 16", "≤ 200", "H100-h"] };
  $("#tierTable").innerHTML = `<thead><tr><th>Resource tier</th><th>Agent development</th><th>Per verification</th><th>Reference precompute</th><th>Cases</th></tr></thead><tbody>${Object.entries(caps).map(([k, v]) =>
    `<tr><td><span class="tag ${k === "pro" ? "red" : ""}">${k}</span></td><td class="mono">${v[0]}</td><td class="mono">${v[1]}</td><td class="mono">${v[2]}</td><td class="mono">${allCases.filter(c => c.budget === k).length}</td></tr>`).join("")}
    <tr><td colspan="5" class="panel-note">Units: low / medium / high are GPU-h on one 24GB reference card, pro is H100-h. CPU-only tasks count as low.</td></tr></tbody>`;
  (() => {
    const box = $("#matrix");
    const cols = ["low", "medium", "high", "pro"];
    const count = (d, r) => allCases.filter(c => c.difficulty === d && c.budget === r).length;
    const max = Math.max(...LEVELS.flatMap(d => cols.map(r => count(d, r))));
    let html = `<span></span>${cols.map(c => `<span class="mh">resources ${c}</span>`).join("")}`;
    LEVELS.forEach(d => {
      html += `<span class="ml" title="${esc(IDX.legend.difficulty[d])}">difficulty ${d}</span>`;
      cols.forEach(r => {
        const n = count(d, r);
        const pct = n ? 12 + 60 * n / max : 0;
        html += `<button type="button" class="${n ? "" : "zero"}" data-d="${d}" data-r="${r}" style="background:color-mix(in srgb, var(--blue) ${pct}%, var(--sheet))${n / max > 0.6 ? ";color:var(--sheet)" : ""}" ${n ? "" : "disabled"} aria-label="difficulty ${d}, resources ${r}: ${n} cases">${n}</button>`;
      });
    });
    box.innerHTML = html;
    box.addEventListener("click", e => { const b = e.target.closest("button[data-d]"); if (!b) return; Cases.set({ diff: b.dataset.d, res: b.dataset.r, group: null, status: null, kind: null, q: "" }); document.getElementById("cases").scrollIntoView({ behavior: RM ? "auto" : "smooth" }); });
    onView(box, () => [...box.querySelectorAll("button")].forEach((b, i) => pop(b, i * 50, 320)));
  })();

  // ------------------------------------------------------------------ standard cases (task statement v2.1)
  (() => {
    const STD = A.standard || [];
    if (!STD.length) { document.getElementById("standard")?.remove(); return; }
    const byMode = m => STD.filter(r => r.mode === m);
    $("#stdModes").innerHTML = ["checkpoint", "judge_trains", "no_training"].map(m => {
      const rows = byMode(m);
      return `<div class="std-mode"><h3><span>${MODE[m].name}</span><span class="n">${rows.length}</span></h3>
        <p><b>Submitted:</b> <span class="mono">${esc(MODE[m].what)}</span></p><p>${esc(MODE[m].does)}</p>
        <div class="ids">${rows.map(r => `<button type="button" data-open="${r.id}">${r.id}</button>`).join("")}</div></div>`;
    }).join("");
    const FLOW = [["Retrieve", "Study the anchor and search related work before the cutoff with the offline paper tool (logged)."],
                  ["Synthesise", "What existing methods do, where they fall short, the gap addressed: report Section 2, Related Work."],
                  ["Design", "At least one precise change relative to the anchor, each listed with its code location in <ol id=\"changes\">."],
                  ["Develop and verify", "Train in the sandbox (from interface/baseline/ if useful) and run selfcheck.py: the evaluator's own stages and metrics on a dev split."],
                  ["Submit", "One shot: code (+ checkpoint and train.py), design report, references.yaml."]];
    $("#stdFlow").innerHTML = FLOW.map(([a, b]) => `<li><b>${esc(a)}</b><span>${esc(b)}</span></li>`).join("");
    const gate = g => g.min != null ? `${g.metric} ≥ ${g.min}` : `${g.metric} ≤ ${g.max}`;
    let filt = null;
    function table() {
      const rows = STD.filter(r => !filt || r.mode === filt);
      $("#stdTable").innerHTML = `<thead><tr><th>Case</th><th>Track</th><th>Mode</th><th>Submitted</th><th>Primary metric</th><th>Gates</th><th>Hardware</th><th>Runs</th><th>Plumbing</th></tr></thead><tbody>${rows.map(r =>
        `<tr class="row" data-open="${r.id}"><td class="mono nowrap">${r.id}</td><td>${esc(r.track || "")}</td><td><span class="tag ${MODE[r.mode]?.tag || "plain"}">${esc(MODE[r.mode]?.name || r.mode || "")}</span></td>
          <td class="mono">${esc([r.entry, ...(r.kind === "code_plus_checkpoint" ? ["checkpoint/"] : []), ...(r.required_files || [])].filter(Boolean).join(" + "))}</td>
          <td class="mono">${esc(r.primary ? `${r.primary.metric} (${r.primary.goal})` : "")}</td>
          <td class="std-gates">${r.gates && r.gates.length ? r.gates.map(g => esc(gate(g))).join("<br>") : "none (validity only)"}</td>
          <td>${esc((r.hardware || "").toUpperCase())}</td><td class="mono">${r.runs ?? "—"}</td>
          <td>${r.verified_e2e_synthetic ? `<span class="tag green">end to end (synthetic)</span>` : `<span class="tag amber">needs GPU/torch</span>`}</td></tr>`).join("")}</tbody>`;
    }
    const seg = $("#stdFilter");
    seg.innerHTML = [null, "checkpoint", "judge_trains", "no_training"].map(m => `<button type="button" role="tab" data-m="${m || ""}" aria-selected="${m === filt}">${m ? MODE[m].name : "All"}</button>`).join("");
    seg.addEventListener("click", e => { const b = e.target.closest("button[data-m]"); if (!b) return; filt = b.dataset.m || null; seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-selected", x === b)); table(); });
    table();
    const e2e = STD.filter(r => r.verified_e2e_synthetic).length;
    const waiting = STD.filter(r => r.activation === "awaiting_data").length;
    $("#stdNote").textContent = `${e2e} of ${STD.length} cases run end to end in tests on synthetic data (reference train.py → checkpoint → stage pipeline → metrics); the others need PyTorch, a GPU or a simulator and have not been executed yet. ${waiting} of ${STD.length} wait for the operator to freeze the real data; until then no result is scorable. Click a row for the full task statement.`;
    const L = A.launch || {};
    const LS = { preflight: ["Preflight", "packages; the runner's exact claude -p argv with an isolated HOME and the prompt on stdin"],
                 key: ["Key", "hidden-set key outside the repository, 0600"],
                 data: ["Data", "download from the pinned source and freeze: public files, encrypted held-out split, hash lock"],
                 mock: ["Mock agent", "a scripted agent walks the whole Workflow through the real runner, no tokens"],
                 agent: ["Agent", "claude -p in the sandbox; full trajectory, retrieval log and submission recorded"],
                 judge: ["Judge", "calibration runs of the agent's and the reference submission (never scorable)"] };
    $("#stdLaunch").innerHTML = (L.steps || []).map(s => `<li class="ok"><b>${esc(LS[s]?.[0] || s)}</b><span>${esc(LS[s]?.[1] || "")}</span></li>`).join("");
    $("#stdLaunchNote").innerHTML = `<span class="mono">${esc(L.script || "")}</span>: verified — ${esc(L.verified || "")}. Pending — ${esc(L.pending || "")}.`;
    document.getElementById("standard").addEventListener("click", e => { const b = e.target.closest("[data-open]"); if (b) Drawer.open(b.dataset.open, b); });
  })();

  // ------------------------------------------------------------------ case browser
  const Cases = (() => {
    const st = { group: null, diff: null, res: null, status: null, kind: null, q: "" };
    const nGroups = new Set(allCases.map(c => c.group)).size;
    $("#casesLede").textContent = `${nGroups} groups, ${IDX.tracks.length} tracks, ${allCases.length} cases. Click a row to see the task, verification plan and evaluation mode.`;
    function chips(id, key, values, names) {
      const box = $(id);
      box.querySelectorAll(".chip").forEach(x => x.remove());
      [null, ...values].forEach(v => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "chip"; b.textContent = v == null ? "All" : names[v];
        b.setAttribute("aria-pressed", st[key] === v);
        b.addEventListener("click", () => { st[key] = v; render(); });
        box.appendChild(b);
      });
    }
    const search = $("#caseSearch");
    search.addEventListener("input", () => { st.q = search.value; renderList(); });
    function hit(c) {
      const q = st.q.trim().toLowerCase();
      return (!st.group || c.group === st.group) && (!st.diff || c.difficulty === st.diff) && (!st.res || c.budget === st.res) && (!st.status || c.status === st.status) && (!st.kind || c.oj?.kind === st.kind)
        && (!q || [c.id, c.anchor, c.arxiv, c.task, c.metrics, c.track.name].join(" ").toLowerCase().includes(q));
    }
    function renderList() {
      const list = $("#caseList");
      let html = "", n = 0;
      GROUP_ORDER.forEach(g => {
        const rows = allCases.filter(c => c.group === g && hit(c));
        if (!rows.length) return;
        n += rows.length;
        html += `<div class="case-group-h"><h3>${GROUP_TITLE[g]}</h3><span>${esc(IDX.legend.groups[g])}</span></div>`;
        html += rows.map(c => `<button type="button" class="case-row" data-id="${c.id}"><span class="cid">${c.id}</span><span class="canchor">${esc(c.anchor)}</span><span class="ctask">${esc(c.task)}</span>
          <span class="cmeta"><span class="tag ${c.status === "built" ? "built" : "plain"}">${STATUS_ZH[c.status] || c.status || "status unknown"}</span><span class="tag plain">${READINESS_ZH[c.readiness] || c.readiness || "readiness unknown"}</span><span class="tag plain">${KIND_ZH[c.oj?.kind] || "submission TBD"}</span>${c.oj?.standard ? `<span class="tag green">standard case</span>` : (c.oj?.judge && c.oj.judge !== "pending" ? `<span class="tag green">judge wired in</span>` : "")}<span class="tag plain">difficulty ${c.difficulty}</span><span class="tag ${c.budget === "pro" ? "red" : ""}">resources ${c.budget}</span></span></button>`).join("");
      });
      list.innerHTML = html || `<div class="case-empty">No cases match. Try another filter or clear the search.</div>`;
      $("#caseCount").textContent = `Showing ${n} / ${allCases.length}`;
    }
    function render() {
      chips("#fGroup", "group", GROUP_ORDER, GROUP_TITLE);
      chips("#fDiff", "diff", LEVELS, { low: "low", medium: "medium", high: "high" });
      chips("#fRes", "res", ["low", "medium", "high", "pro"], LEVEL_ZH);
      chips("#fStatus", "status", ["built", "spec", "scaffold", "planned"], STATUS_ZH);
      chips("#fKind", "kind", ["code_only", "code_plus_checkpoint", "report_only"], KIND_ZH);
      search.value = st.q;
      renderList();
    }
    $("#caseList").addEventListener("click", e => { const b = e.target.closest(".case-row"); if (b) Drawer.open(b.dataset.id, b); });
    render();
    return { set(p) { Object.assign(st, p); render(); } };
  })();

  // ------------------------------------------------------------------ detail drawer
  const Drawer = (() => {
    const dr = $("#drawer"), scrim = $("#scrim"), body = $("#drawerBody");
    let opener = null;
    const PX = IDX.legend.proxies;
    function modeOf(c) {
      const px = c.verify_plan?.proxy || [];
      if (c.group === "pro") return "pro";
      if (px.length && px.every(p => p === "JUDGE")) return "judge";
      if (px[0] === "JUDGE") return "judge";
      return "budget";
    }
    function flowFor(c) {
      const v = c.verify_plan || {};
      const u = v.unit || "GPU-h";
      const m = modeOf(c);
      if (m === "pro") return [["4 × H100, 24 hours", "development, debugging and final training all happen within these 24 hours"], ["Deliver code, weights and design report", "logs, GPU telemetry and the checkpoint hash chain are written by the harness"], ["Judge verification about " + v.verify + " " + u, "outside the sandbox, the weights are evaluated on the hidden set; one randomly chosen registered point is resumed for 1 hour to check"], ["Compared with strong industry recipes", "a lead counts only above the 2σ noise floor"], ["Scoring", "M · D · Q · (1 + 0.25X); engineering metrics such as MFU and throughput reported separately"]];
      if (m === "judge") return [["Given a published claim and a paper snapshot", "material from before the cutoff, same retrieval rules as other cases"], ["Submit a judgment report", "conclusion, reasons and a minimal verification plan; no experiments"], ["Judge check", "verify the judgment with existing reference results or a small experiment"], ["Scoring", "correctness of the judgment + information content of the verification plan"]];
      const k = c.oj?.kind;
      if (c.oj?.standard && c.oj?.mode) {
        const md = MODE[c.oj.mode];
        return [["Retrieve and synthesise", "offline paper search before the cutoff; Related Work in the report"],
                ["Design and develop in the sandbox", "changes relative to the anchor with code locations; selfcheck.py runs the evaluator's own pipeline on a dev split"],
                ["One-shot submission", md.what + ", design report, references"],
                ["Evaluation: " + md.name, md.does],
                ["Result", "metrics from the case's metrics.py; valid only if every gate holds"]];
      }
      return [["Agent development " + (v.cpu ? "CPU only" : v.dev + " " + u), "trial and error in its own workspace"], ["One-shot submission", (k === "code_plus_checkpoint" ? "model code and checkpoint" : "method code") + ", design report, references"], ["OJ evaluation about " + (v.cpu ? "CPU" : v.verify + " " + u), KIND_READS[k] || "run on public and hidden settings"], ["Three thresholds", "interface check, staying at SOTA, differentiation"], ["Scoring", "M · D · Q · (1 + 0.25X)"]];
    }
    function ojSec(c) {
      const o = c.oj || {};
      if (!o.kind) return "";
      const judge = o.judge && o.judge !== "pending";
      return `<div class="dr-sec"><span class="k">OJ configuration</span>
        <div class="dr-tags"><span class="tag">${esc(KIND_ZH[o.kind])}</span>${o.entry ? `<span class="tag plain mono">${esc(o.entry)}</span>` : ""}${o.checkpoint_dir ? `<span class="tag plain mono">${esc(o.checkpoint_dir)}/</span>` : ""}<span class="tag plain">${o.report_profile === "judgment_report_v2" ? "judgment report" : "design report"}</span><span class="tag ${judge ? "green" : "amber"}">${JUDGE_ZH[o.judge] || o.judge}</span>${o.standard ? `<span class="tag green">standard case</span>` : ""}${o.mode ? `<span class="tag ${MODE[o.mode]?.tag || "plain"}">${esc(MODE[o.mode]?.name || o.mode)}</span>` : ""}${(o.required_files || []).map(f => `<span class="tag plain mono">${esc(f)}</span>`).join("")}${o.activation === "awaiting_data" ? `<span class="tag amber">awaiting frozen data</span>` : ""}${o.handcrafted ? `<span class="tag plain">handcrafted</span>` : ""}</div>
        <p>${esc((o.mode && MODE[o.mode] ? MODE[o.mode].does : (KIND_READS[o.kind] || "")).replace(/\.$/, ""))}${judge ? `; metrics ${esc(o.judge_metrics.join(", "))}${o.hidden_rows ? `, hidden set ${o.hidden_rows} rows (encrypted)` : ""}` : "; format checks only until the judge is frozen"}.</p></div>
        <div class="dr-sec"><span class="k">Agent view (read-only)</span><ul class="tree small">${(o.view || []).map(f => `<li class="f"><span class="mono">${esc(f)}</span></li>`).join("")}</ul>
        ${o.task_md ? `<details class="taskmd"><summary>Show the task.md the agent receives</summary><pre>${esc(o.task_md)}</pre></details>` : ""}</div>`;
    }
    function open(id, from) {
      const c = allCases.find(x => x.id === id); if (!c) return;
      opener = from || null;
      const v = c.verify_plan || {};
      const u = v.unit || "GPU-h";
      const link = c.arxiv ? `<a href="https://arxiv.org/abs/${esc(c.arxiv)}" target="_blank" rel="noopener">arXiv:${esc(c.arxiv)}</a>` : c.link ? `<a href="${esc(c.link)}" target="_blank" rel="noopener">${esc(c.venue || "link")}</a>` : "anchor paper to be chosen";
      const mode = modeOf(c);
      const modeName = c.oj?.standard && c.oj?.mode ? `${MODE[c.oj.mode]?.name || c.oj.mode} (standard case)` : { budget: "Budget micro-scale retraining", pro: "Pro audit", judge: "judgment" }[mode];
      body.innerHTML = `
        <div class="dr-sec"><span class="dr-id">${c.id} · ${esc(GROUP_TITLE[c.group])} · ${esc(c.track.name)}</span><h2 class="dr-title" id="drawerTitle">${esc(c.anchor)}</h2><span>${link}</span>
          <div class="dr-tags">${c.status === "built" ? `<span class="tag green">built</span>` : `<span class="tag plain">${STATUS_ZH[c.status] || c.status || "status unknown"}</span>`}<span class="tag plain">${READINESS_ZH[c.readiness] || c.readiness || "readiness unknown"}</span><span class="tag plain">difficulty ${c.difficulty}</span><span class="tag ${c.budget === "pro" ? "red" : ""}">resources ${c.budget}</span><span class="tag plain">${SUBSET_ZH[c.subset] || c.subset}</span><span class="tag plain">${c.type}</span>${c.verify ? `<span class="tag amber">anchor to be verified</span>` : ""}</div></div>
        <div class="dr-sec"><span class="k">Task</span><p>${esc(c.task)}</p></div>
        ${ojSec(c)}
        <div class="dr-sec"><span class="k">Metrics</span><p>${esc(c.metrics)}</p></div>
        <div class="dr-sec"><span class="k">Evaluation mode: ${modeName}</span><div class="dr-flow">${flowFor(c).map(([a, b]) => `<div class="fl"><span class="d"></span><span class="c"><b>${esc(a)}</b><span>${esc(b)}</span></span></div>`).join("")}</div></div>
        <div class="dr-sec"><span class="k">Reduced-scale verification plan</span>
          <div class="dr-tags">${(v.proxy || []).map(p => `<span class="tag" title="${esc(PX[p] || p)}">${p} ${esc(PX[p] || "")}</span>`).join("")}${(v.risk || []).map(r => `<span class="tag amber">${esc(r)}</span>`).join("")}</div>
          <p>${esc(v.plan || "")}</p>
          <div class="dr-nums">${v.cpu ? `<div><span class="n">CPU</span><span class="l">CPU-only task</span></div>` : `<div><span class="n">${v.dev}</span><span class="l">development ${u}</span></div><div><span class="n">${v.verify}</span><span class="l">per verification ${u}</span></div><div><span class="n">${v.ref}</span><span class="l">reference precompute ${u}</span></div>`}</div></div>
        <div class="dr-sec"><span class="k">Difficulty and resources</span><p>Difficulty ${c.difficulty}: ${esc(IDX.legend.difficulty[c.difficulty])}</p><p>Resources ${c.budget}: ${esc(IDX.legend.budget_tiers[c.budget])}</p></div>
        ${c.pairs_with ? `<div class="dr-sec"><span class="k">Pairing</span><p>Paired with <button class="chip" data-open="${c.pairs_with}">${c.pairs_with}</button> of the Budget edition; the agreement of their rankings tests the fidelity of the micro-scale proxy.</p></div>` : ""}
        <div class="dr-sec"><span class="k">Track: ${esc(c.track.name)}</span><p>${esc(c.track.why_focus)}</p><p class="panel-note">Anti-gaming: ${esc((c.track.anti_gaming || []).join("; "))}</p></div>
        ${c.id === "B-OPT-001" ? `<div class="dr-sec"><a class="btn small primary" href="#flow" data-close>Watch the full replay</a></div>` : ""}`;
      dr.classList.add("open"); dr.setAttribute("aria-hidden", "false"); scrim.hidden = false;
      dr.scrollTop = 0;
      $("#drawerClose").focus({ preventScroll: true });
      [...body.querySelectorAll(".dr-sec")].forEach((s, i) => fade(s, 80 + i * 50, 280, 4));
    }
    function close() {
      dr.classList.remove("open"); dr.setAttribute("aria-hidden", "true"); scrim.hidden = true;
      if (opener && document.body.contains(opener)) opener.focus({ preventScroll: true });
    }
    $("#drawerClose").addEventListener("click", close);
    scrim.addEventListener("click", close);
    document.addEventListener("keydown", e => { if (e.key === "Escape" && dr.classList.contains("open")) close(); });
    body.addEventListener("click", e => {
      const o = e.target.closest("[data-open]"); if (o) { open(o.dataset.open, opener); return; }
      if (e.target.closest("[data-close]")) close();
    });
    return { open };
  })();

  // ------------------------------------------------------------------ scale
  (() => {
    const b = allCases.filter(c => c.verify_plan && !c.verify_plan.unit), p = allCases.filter(c => c.verify_plan && c.verify_plan.unit);
    const sum = (arr, k) => arr.reduce((a, c) => a + (c.verify_plan[k] || 0), 0);
    const stats = [
      [allCases.length, "cases", `${IDX.tracks.length} tracks in 5 groups`],
      [Math.round(sum(b, "ref")), "GPU-h", "Budget edition reference precompute (one-off)"],
      [Math.round(sum(b, "dev") + sum(b, "verify")), "GPU-h", "one agent through the Budget edition (development + verification)"],
      [Math.round(sum(p, "ref")), "H100-h", "Pro Set reference recipes (one-off)"],
      [Math.round(sum(p, "dev") + sum(p, "verify")), "H100-h", "one agent through the Pro Set"],
      [allCases.filter(c => c.verify_plan?.risk?.length).length, "cases", "still carry residual risk and depend on fidelity checks"]
    ];
    const box = $("#scaleGrid");
    box.innerHTML = stats.map(s => `<div class="stat"><span class="n" data-to="${s[0]}">${s[0].toLocaleString()}</span><span class="u">${s[1]}</span><span class="l">${s[2]}</span></div>`).join("");
    onView(box, () => box.querySelectorAll(".n").forEach(n => { const to = +n.dataset.to; tween(1100, t => { n.textContent = Math.round(to * t).toLocaleString(); }); }));
  })();
})();
