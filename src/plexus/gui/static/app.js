"use strict";
/* Plexus spec node-editor — vanilla JS/SVG. The graph is a pure function of the spec:
   operators / sets / fields are nodes, `at`/`to`/`from` and the schedule are the edges.
   Editing mutates the spec model and re-renders; Save round-trips validated YAML. */

const SVGNS = "http://www.w3.org/2000/svg";
const $ = (s, r = document) => r.querySelector(s);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ---- geometry constants ----
const HDR = 26, ROW = 16, PADV = 8, MAXROWS = 6;
const OPW = 220, SETW = 190, FLDW = 172;
const FIELD_X = 40, OP_X = 330, SET_X = 700, COL_GAP = 26;

// ---- global state ----
const S = {
  catalog: null, opByName: {}, aliasTo: {},
  repoRoot: "", specPath: null, specRel: null,
  spec: null,
  layout: { nodes: {}, view: { x: 60, y: 40, k: 1 } },
  view: { x: 60, y: 40, k: 1 },
  sel: null,                 // {type, id}
  media: null,               // {video, poster?, name} if a rendered mp4 sits by the spec
  geom: {},                  // nodeId -> {type,name,x,y,w,h,color,ports,op}
  showSchedule: true,
  dirty: false,
  forceSaveArmed: false,
  validateTimer: null,
};

// ---- tiny dom helpers ----
function svg(tag, attrs = {}, kids = []) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  for (const c of [].concat(kids)) if (c) e.appendChild(c);
  return e;
}
function svtext(x, y, cls, str, anchor) {
  const t = svg("text", { x, y, class: cls }); if (anchor) t.setAttribute("text-anchor", anchor);
  t.textContent = str; return t;
}
function h(tag, props = {}, kids = []) {
  const e = document.createElement(tag);
  for (const k in props) {
    if (k === "class") e.className = props[k];
    else if (k === "html") e.innerHTML = props[k];
    else if (k.startsWith("on") && typeof props[k] === "function") e.addEventListener(k.slice(2), props[k]);
    else if (props[k] != null) e.setAttribute(k, props[k]);
  }
  for (const c of [].concat(kids)) if (c != null) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return e;
}

// ---- api ----
async function apiGet(u) { const r = await fetch(u); return r.json(); }
async function apiPost(u, body) {
  const r = await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

function toast(msg, kind = "") {
  const t = $("#toast"); t.textContent = msg; t.className = "toast " + kind;
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), 2600);
}

// =====================================================================
//  value formatting / coercion
// =====================================================================
function fmtVal(v) {
  if (v == null) return "·";
  if (Array.isArray(v)) return "[" + v.map(fmtVal).join(",") + "]";
  if (typeof v === "number") return (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)) ? v.toExponential(2) : String(+v.toFixed(4)).replace(/\.?0+$/, "") || "0";
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}
function coerce(raw, type) {
  if (type === "bool") return !!raw;
  if (type === "number") { const n = Number(raw); return (raw !== "" && Number.isFinite(n)) ? n : raw; }
  return raw;
}

// =====================================================================
//  catalog
// =====================================================================
function indexCatalog(cat) {
  S.catalog = cat;
  S.opByName = {}; S.aliasTo = cat.by_name || {};
  for (const op of cat.operators) S.opByName[op.name] = op;
}
function opDef(name) {                       // resolve alias -> canonical def (or null)
  if (S.opByName[name]) return S.opByName[name];
  const canon = S.aliasTo[name];
  return canon ? S.opByName[canon] : null;
}
function implOf(op, def) {
  if (!def) return null;
  const nm = op.implementation || def.default_impl;
  return def.implementations[nm] || def.implementations[def.default_impl] || null;
}
function paramRows(op, def) {                 // [{key,val,required,missing,type,role,options}]
  const impl = implOf(op, def);
  const rows = []; const seen = new Set();
  if (impl) for (const p of impl.params) {
    seen.add(p.name);
    const has = op[p.name] !== undefined;
    rows.push({ key: p.name, val: has ? op[p.name] : p.default, required: p.required,
                missing: p.required && !has, type: p.type, role: p.role, options: p.options });
  }
  for (const k in op) {                       // extras present on the op but not in schema
    if (["op", "at", "to", "from", "implementation", "emit"].includes(k)) continue;
    if (seen.has(k)) continue;
    rows.push({ key: k, val: op[k], required: false, missing: false, type: typeof op[k] === "boolean" ? "bool" : typeof op[k] === "number" ? "number" : "str" });
  }
  return rows;
}
function opColor(op) { const d = opDef(op.op); return d ? d.color : "#8b93a1"; }
function opKind(op) { const d = opDef(op.op); return d ? d.kind : ""; }

// =====================================================================
//  spec model helpers
// =====================================================================
function setNames() { return Object.keys(S.spec.sets || {}); }
function fieldNames() { return Object.keys(S.spec.fields || {}); }
function blockDecl(decl) {
  if (decl && typeof decl === "object") return { width: +(decl.width ?? 1), integration: decl.integration || "first_order", boundary: decl.boundary || "free" };
  return { width: +(decl || 1), integration: "first_order", boundary: "free" };
}
function markDirty() {
  S.dirty = true; S.forceSaveArmed = false;
  $("#btnSave").textContent = "SAVE *";
  scheduleValidate();
}
function scheduleValidate() {
  clearTimeout(S.validateTimer);
  S.validateTimer = setTimeout(runValidate, 450);
}

// =====================================================================
//  geometry / layout
// =====================================================================
function nodeId(type, name) { return type + ":" + name; }

function opPorts(op, w, hh) {
  const def = opDef(op.op), kind = def ? def.kind : "";
  const ports = { act: { x: w, y: HDR / 2, face: 1, cls: "act", label: "" } };
  const exch = kind === "exchange" || op.to !== undefined || op.from !== undefined;
  if (exch) {
    // input on the LEFT (reads a field), output on the RIGHT (writes a field)
    ports.from = { x: 0, y: hh - 20, face: -1, cls: "field", label: "in" };
    ports.to   = { x: w, y: hh - 20, face: 1,  cls: "field", label: "out" };
  }
  ports.schedout = { x: w / 2, y: hh, face: 0, cls: "sched", hidden: true };
  ports.schedin  = { x: w / 2, y: 0, face: 0, cls: "sched", hidden: true };
  return ports;
}

function buildGeom() {
  S.geom = {};
  const L = S.layout.nodes || {};
  // operators, ordered by schedule then remainder
  const ops = S.spec.operators || [];
  const orderIdx = {}; let oi = 0;
  for (const st of (S.spec.schedule || [])) if (typeof st === "string" && orderIdx[st] === undefined) orderIdx[st] = oi++;
  const opsOrdered = ops.slice().sort((a, b) => (orderIdx[a.op] ?? 999) - (orderIdx[b.op] ?? 999));

  let y = 40;
  for (const op of opsOrdered) {
    const def = opDef(op.op);
    const rows = paramRows(op, def);
    const shown = Math.min(rows.length, MAXROWS);
    const multi = def && Object.keys(def.implementations).length > 1;
    const lines = (multi ? 1 : 0) + Math.max(shown, 0) + (rows.length > MAXROWS ? 1 : 0);
    const hh = HDR + 8 + Math.max(lines, 1) * ROW + PADV;
    const id = nodeId("op", op.op);
    const pos = L[id] || { x: OP_X, y };
    S.geom[id] = { type: "op", name: op.op, op, x: pos.x, y: pos.y, w: OPW, h: hh,
                   color: opColor(op), ports: opPorts(op, OPW, hh), rows, shown, multi };
    y += hh + COL_GAP;
  }
  // sets
  let ys = 40;
  for (const name of setNames()) {
    const s = S.spec.sets[name] || {};
    const blocks = s.state ? Object.keys(s.state) : [];
    const lines = 1 + Math.max(blocks.length, 1);
    const hh = HDR + 8 + lines * ROW + PADV;
    const id = nodeId("set", name);
    const pos = L[id] || { x: SET_X, y: ys };
    S.geom[id] = { type: "set", name, x: pos.x, y: pos.y, w: SETW, h: hh, color: S.catalog.set_color,
                   ports: { in: { x: 0, y: HDR / 2, face: -1, cls: "state", label: "" } }, blocks };
    ys += hh + COL_GAP;
  }
  // fields (left column)
  let yf = 40;
  for (const name of fieldNames()) {
    const f = S.spec.fields[name] || {};
    const keys = Object.keys(f);
    const lines = 1 + Math.max(keys.length, 0);
    const hh = HDR + 8 + Math.max(lines, 1) * ROW + PADV;
    const id = nodeId("field", name);
    const pos = L[id] || { x: FIELD_X, y: yf };
    S.geom[id] = { type: "field", name, x: pos.x, y: pos.y, w: FLDW, h: hh, color: S.catalog.field_color,
                   ports: { out: { x: FLDW, y: HDR / 2, face: 1, cls: "field", label: "" } }, keys };
    yf += hh + COL_GAP;
  }
}

function portAbs(id, port) {
  const g = S.geom[id]; if (!g || !g.ports[port]) return null;
  const p = g.ports[port]; return { x: g.x + p.x, y: g.y + p.y, face: p.face };
}

// =====================================================================
//  edges
// =====================================================================
function computeEdges() {
  const E = [];
  for (const op of (S.spec.operators || [])) {
    const oid = nodeId("op", op.op);
    if (op.at) {
      const tgt = S.geom[nodeId("set", op.at)] ? nodeId("set", op.at)
                : S.geom[nodeId("field", op.at)] ? nodeId("field", op.at) : null;
      if (tgt) E.push({ cls: "act", a: [oid, "act"], b: [tgt, S.geom[tgt].type === "set" ? "in" : "out"], op: op.op });
    }
    if (op.from && S.geom[nodeId("field", op.from)]) E.push({ cls: "from", a: [nodeId("field", op.from), "out"], b: [oid, "from"], op: op.op });
    if (op.to && S.geom[nodeId("field", op.to)]) E.push({ cls: "to", a: [oid, "to"], b: [nodeId("field", op.to), "out"], op: op.op });
  }
  if (S.showSchedule) {
    const seq = (S.spec.schedule || []).filter(t => typeof t === "string" && S.geom[nodeId("op", t)]);
    for (let i = 0; i < seq.length - 1; i++)
      E.push({ cls: "sched", a: [nodeId("op", seq[i]), "schedout"], b: [nodeId("op", seq[i + 1]), "schedin"], sched: true });
  }
  return E;
}
function edgePath(a, b) {
  const dx = Math.max(40, Math.abs(b.x - a.x) * 0.4);
  const c1x = a.x + dx * (a.face || 0.001), c2x = b.x + dx * (b.face || -0.001);
  return `M ${a.x} ${a.y} C ${c1x} ${a.y} ${c2x} ${b.y} ${b.x} ${b.y}`;
}
function renderEdges() {
  const layer = $("#edges"); layer.textContent = "";
  for (const e of computeEdges()) {
    const a = portAbs(e.a[0], e.a[1]), b = portAbs(e.b[0], e.b[1]);
    if (!a || !b) continue;
    const d = edgePath(a, b);
    layer.appendChild(svg("path", { class: "edge " + e.cls, d }));
  }
}

// =====================================================================
//  nodes
// =====================================================================
function renderNodes() {
  const layer = $("#nodes"); layer.textContent = "";
  for (const id in S.geom) layer.appendChild(renderNode(id, S.geom[id]));
}
function renderNode(id, g) {
  const selCls = (S.sel && nodeId(S.sel.type, S.sel.id) === id) ? " sel" : "";
  const grp = svg("g", { class: "node" + selCls, "data-nodeid": id, transform: `translate(${g.x},${g.y})` });
  grp.appendChild(svg("rect", { class: "box", x: 0, y: 0, width: g.w, height: g.h, rx: 2 }));
  grp.appendChild(svg("rect", { class: "hdrbar", x: 0, y: 0, width: g.w, height: HDR }));
  grp.appendChild(svg("rect", { x: 0, y: 0, width: 3, height: g.h, fill: g.color }));
  grp.appendChild(svtext(10, 17, "title", g.name));

  if (g.type === "op") {
    const kind = opKind(g.op);
    if (kind) grp.appendChild(svtext(g.w - 8, 17, "kindbadge", kind, "end"));
    let yy = HDR + 14;
    if (g.multi) { const def = opDef(g.op.op); const im = g.op.implementation || def.default_impl;
      grp.appendChild(svtext(10, yy, "impl", "▸ " + im)); yy += ROW; }
    let n = 0;
    for (const r of g.rows) {
      if (n >= MAXROWS) { grp.appendChild(svtext(10, yy, "meta", `+${g.rows.length - MAXROWS} more…`)); break; }
      grp.appendChild(svtext(10, yy, "pkey", r.key));
      grp.appendChild(svtext(g.w - 8, yy, "pval" + (r.missing ? " req-missing" : ""), r.missing ? "required" : fmtVal(r.val), "end"));
      yy += ROW; n++;
    }
  } else if (g.type === "set") {
    const s = S.spec.sets[g.name] || {};
    grp.appendChild(svtext(g.w - 8, 17, "kindbadge", "set", "end"));
    let yy = HDR + 14;
    grp.appendChild(svtext(10, yy, "meta", "n=" + (s.n ?? "?") + (s.buffer ? "  buf=" + s.buffer : ""))); yy += ROW;
    if (g.blocks.length) for (const bn of g.blocks) {
      const d = blockDecl(s.state[bn]);
      const row = svg("g", { class: "srow" });
      row.appendChild(svtext(10, yy, "sname", bn));
      row.appendChild(svtext(g.w - 8, yy, "smeta", "w" + d.width + "·" + d.integration.replace("_order", ""), "end"));
      grp.appendChild(row); yy += ROW;
    } else { grp.appendChild(svtext(10, yy, "meta", "default state (pos·vel)")); }
  } else {
    const f = S.spec.fields[g.name] || {};
    grp.appendChild(svtext(g.w - 8, 17, "kindbadge", "field", "end"));
    let yy = HDR + 14;
    grp.appendChild(svtext(10, yy, "meta", f.kind ? "kind=" + f.kind : "field")); yy += ROW;
    for (const k of g.keys) { if (k === "kind") continue; grp.appendChild(svtext(10, yy, "meta", k + "=" + fmtVal(f[k]))); yy += ROW; }
  }

  // drag grip over header (below text, which is pointer-events:none)
  grp.appendChild(svg("rect", { class: "hdr-grip", x: 0, y: 0, width: g.w, height: HDR }));
  // ports
  for (const pn in g.ports) {
    const p = g.ports[pn]; if (p.hidden) continue;
    const c = svg("circle", { class: "port " + p.cls, cx: p.x, cy: p.y, r: 4, "data-port": pn, "data-nodeid": id });
    grp.appendChild(c);
    if (p.label) grp.appendChild(svtext(p.face < 0 ? p.x + 8 : p.x - 8, p.y + 3, "portlabel", p.label, p.face < 0 ? "start" : "end"));
  }
  return grp;
}

// =====================================================================
//  render top-level
// =====================================================================
function applyView() { $("#world").setAttribute("transform", `translate(${S.view.x},${S.view.y}) scale(${S.view.k})`); }
function render() {
  $("#emptyState").classList.toggle("hidden", !!S.spec);
  if (!S.spec) { $("#nodes").textContent = ""; $("#edges").textContent = ""; renderInspector(); renderSchedule(); return; }
  buildGeom(); renderNodes(); renderEdges(); applyView();
  renderInspector(); renderSchedule();
  const c = S.catalog.counts;
  $("#counts").textContent = `${(S.spec.operators||[]).length} ops · ${setNames().length} sets · ${fieldNames().length} fields  |  atlas: ${c.canonical} ops`;
}

// =====================================================================
//  palette
// =====================================================================
function renderPalette() {
  const list = $("#palList"); list.textContent = "";
  const byFam = {};
  for (const op of S.catalog.operators) (byFam[op.family] = byFam[op.family] || []).push(op);
  const q = ($("#palFilter").value || "").toLowerCase();
  $("#palCount").textContent = "· " + S.catalog.operators.length;
  for (const fam of S.catalog.families) {
    const ops = (byFam[fam.name] || []).filter(o => !q || o.name.includes(q) || (o.mechanism_tags || []).some(t => t.includes(q)));
    if (!ops.length) continue;
    const wrap = h("div", { class: "pal-fam" }, [
      h("div", { class: "pal-fam-h" }, [h("span", { class: "swatch", style: `background:${fam.color}` }), fam.name]),
    ]);
    for (const op of ops) {
      const el = h("div", { class: "pal-op", draggable: "true", title: (op.mechanism_tags || []).join(", ") || op.name,
        style: `border-left-color:${op.color}` }, [
        h("span", { class: "nm" }, [op.name]),
        h("span", { class: "kd" }, [op.kind]),
      ]);
      el.addEventListener("dragstart", ev => ev.dataTransfer.setData("text/plain", op.name));
      el.addEventListener("click", () => addOpAtCenter(op.name));
      wrap.appendChild(el);
    }
    list.appendChild(wrap);
  }
}

// =====================================================================
//  inspector
// =====================================================================
function renderInspector() {
  const body = $("#inspBody"); body.textContent = "";
  if (!S.spec) { body.appendChild(h("div", { class: "dim pad" }, ["no spec loaded"])); return; }
  if (!S.sel) { body.appendChild(inspGeneral()); return; }
  if (S.sel.type === "general") body.appendChild(inspGeneral());
  else if (S.sel.type === "op") body.appendChild(inspOp(S.sel.id));
  else if (S.sel.type === "set") body.appendChild(inspSet(S.sel.id));
  else if (S.sel.type === "field") body.appendChild(inspField(S.sel.id));
}

function fieldRow(label, control, opts = {}) {
  return h("div", { class: "field-row" + (opts.req ? " req" : "") }, [
    h("label", { title: opts.role || "" }, [label]),
    h("div", { class: "grow" }, [control].concat(opts.role ? [h("div", { class: "role" }, [opts.role])] : [])),
  ]);
}
function numInput(val, on) {
  const i = h("input", { class: "in", type: "text", value: val ?? "" });
  i.addEventListener("change", () => on(i.value));
  return i;
}
function selInput(val, options, on, blankLabel) {
  const s = h("select", { class: "in" });
  if (blankLabel != null) s.appendChild(h("option", { value: "" }, [blankLabel]));
  for (const o of options) { const opt = h("option", { value: o }, [o]); if (o === val) opt.setAttribute("selected", ""); s.appendChild(opt); }
  s.value = val ?? "";
  s.addEventListener("change", () => on(s.value));
  return s;
}
function boolInput(val, on) {
  const c = h("input", { type: "checkbox" }); c.checked = !!val;
  c.addEventListener("change", () => on(c.checked));
  return h("label", { class: "checkbox" }, [c, h("span", { class: "dim" }, [val ? "true" : "false"])]);
}

function inspGeneral() {
  const g = S.spec.general = S.spec.general || {};
  const wrap = h("div", {});
  wrap.appendChild(h("div", { class: "insp-head" }, [h("span", { class: "nm" }, ["general"]), h("span", { class: "badge" }, ["run"])]));
  wrap.appendChild(fieldRow("name", numInput(g.name, v => { g.name = v; markDirty(); $("#specPath").textContent = (S.specRel || v); })));
  wrap.appendChild(fieldRow("seed", numInput(g.seed, v => { g.seed = coerce(v, "number"); markDirty(); })));
  wrap.appendChild(fieldRow("n_frames", numInput(g.n_frames, v => { g.n_frames = coerce(v, "number"); markDirty(); })));
  wrap.appendChild(fieldRow("dt", numInput(g.dt, v => { g.dt = coerce(v, "number"); markDirty(); })));
  wrap.appendChild(fieldRow("dim", selInput(String(g.dim ?? 2), ["2", "3"], v => { setDim(+v); }), {}));
  wrap.appendChild(fieldRow("boundary", selInput(g.boundary || "wall", S.catalog.boundaries, v => { g.boundary = v; markDirty(); })));
  // world box
  const dim = +(g.dim ?? 2);
  let world = Array.isArray(g.world) ? g.world.slice() : [g.world ?? 1.0];
  while (world.length < dim) world.push(1.0); world = world.slice(0, dim);
  g.world = world;
  const box = h("div", { class: "grow", style: "display:flex;gap:6px" });
  world.forEach((w, i) => {
    const inp = h("input", { class: "in", type: "text", value: w });
    inp.addEventListener("change", () => { g.world[i] = coerce(inp.value, "number"); markDirty(); });
    box.appendChild(inp);
  });
  wrap.appendChild(h("div", { class: "field-row" }, [h("label", {}, ["world"]), box]));
  wrap.appendChild(h("div", { class: "sec" }, ["structure"]));
  wrap.appendChild(h("div", { class: "rowbtns" }, [
    h("button", { class: "mini", onclick: addSet }, ["+ set"]),
    h("button", { class: "mini", onclick: addField }, ["+ field"]),
  ]));
  return wrap;
}
function setDim(d) {
  S.spec.general.dim = d;
  let w = S.spec.general.world; w = Array.isArray(w) ? w.slice() : [w ?? 1.0];
  while (w.length < d) w.push(1.0); S.spec.general.world = w.slice(0, d);
  markDirty(); render();
}

function inspOp(name) {
  const op = (S.spec.operators || []).find(o => o.op === name);
  if (!op) return h("div", { class: "pad dim" }, ["missing"]);
  const def = opDef(name);
  const wrap = h("div", {});
  const head = h("div", { class: "insp-head" }, [h("span", { class: "nm" }, [op.op])]);
  if (def) {
    head.appendChild(h("span", { class: "badge kind", style: `background:${def.color}` }, [def.kind]));
    head.appendChild(h("span", { class: "badge" }, [def.family]));
    const impl = implOf(op, def);
    if (impl && impl.emit) head.appendChild(h("span", { class: "badge" }, ["emit:" + impl.emit]));
  } else head.appendChild(h("span", { class: "badge" }, ["unregistered"]));
  wrap.appendChild(head);
  if (def && def.mechanism_tags && def.mechanism_tags.length)
    wrap.appendChild(h("div", { class: "chips" }, def.mechanism_tags.map(t => h("span", { class: "chip" }, [t]))));

  // implementation selector
  if (def && Object.keys(def.implementations).length > 1) {
    const cur = op.implementation || def.default_impl;
    wrap.appendChild(fieldRow("impl", selInput(cur, Object.keys(def.implementations), v => switchImpl(op, def, v))));
  }
  // acts-on
  const targets = setNames().concat(fieldNames());
  wrap.appendChild(fieldRow("at", selInput(op.at || "", targets, v => { op.at = v; markDirty(); render(); }, "—choose—"), { req: true }));

  // exchange field ports
  const kind = def ? def.kind : "";
  if (kind === "exchange" || op.from !== undefined || op.to !== undefined) {
    wrap.appendChild(fieldRow("from", selInput(op.from || "", fieldNames(), v => { if (v) op.from = v; else delete op.from; markDirty(); render(); }, "—none—")));
    wrap.appendChild(fieldRow("to", selInput(op.to || "", fieldNames(), v => { if (v) op.to = v; else delete op.to; markDirty(); render(); }, "—none—")));
  }

  wrap.appendChild(h("div", { class: "sec" }, ["parameters"]));
  for (const r of paramRows(op, def)) {
    let ctrl;
    if (r.type === "bool") ctrl = boolInput(r.val, v => { op[r.key] = v; markDirty(); render(); });
    else if (r.type === "enum" && r.options) ctrl = selInput(op[r.key] ?? r.val ?? "", r.options, v => { op[r.key] = v; markDirty(); render(); });
    else ctrl = numInput(op[r.key] ?? (r.val ?? ""), v => {
      if (v === "" && !r.required) delete op[r.key]; else op[r.key] = coerce(v, r.type);
      markDirty(); render();
    });
    wrap.appendChild(fieldRow(r.key, ctrl, { req: r.required, role: r.role }));
  }
  wrap.appendChild(h("div", { class: "rowbtns" }, [
    h("button", { class: "mini", onclick: () => addParam(op) }, ["+ param"]),
    h("button", { class: "mini danger", onclick: () => deleteOp(name) }, ["delete op"]),
  ]));
  return wrap;
}
function switchImpl(op, def, impl) {
  const old = implOf(op, def);
  if (impl === def.default_impl) delete op.implementation; else op.implementation = impl;
  // drop params that belonged only to the old impl and aren't in the new one
  const nu = def.implementations[impl];
  const keep = new Set(nu.params.map(p => p.name));
  if (old) for (const p of old.params) if (!keep.has(p.name)) delete op[p.name];
  for (const p of nu.params) if (op[p.name] === undefined && p.default != null) op[p.name] = p.default;
  markDirty(); render();
}
function addParam(op) {
  const k = prompt("param name:"); if (!k) return;
  op[k] = 0; markDirty(); render();
}

function inspSet(name) {
  const s = S.spec.sets[name];
  const wrap = h("div", {});
  wrap.appendChild(h("div", { class: "insp-head" }, [h("span", { class: "nm" }, [name]), h("span", { class: "badge kind", style: `background:${S.catalog.set_color}` }, ["set"])]));
  const nameIn = h("input", { class: "in", type: "text", value: name });
  nameIn.addEventListener("change", () => renameSet(name, nameIn.value));
  wrap.appendChild(fieldRow("name", nameIn));
  wrap.appendChild(fieldRow("n", numInput(s.n, v => { s.n = coerce(v, "number"); markDirty(); render(); })));
  wrap.appendChild(fieldRow("buffer", numInput(s.buffer, v => { if (v === "") delete s.buffer; else s.buffer = coerce(v, "number"); markDirty(); render(); }, )));

  wrap.appendChild(h("div", { class: "sec" }, ["state blocks"]));
  s.state = s.state || {};
  for (const bn of Object.keys(s.state)) wrap.appendChild(stateBlockEditor(s, bn));
  if (!Object.keys(s.state).length) delete s.state;
  wrap.appendChild(h("div", { class: "rowbtns" }, [
    h("button", { class: "mini", onclick: () => addStateBlock(name) }, ["+ block"]),
    h("button", { class: "mini danger", onclick: () => deleteSet(name) }, ["delete set"]),
  ]));
  return wrap;
}
function stateBlockEditor(s, bn) {
  const d = blockDecl(s.state[bn]);
  const box = h("div", { class: "stateblk" });
  const nameIn = h("input", { class: "in", type: "text", value: bn, style: "width:90px" });
  nameIn.addEventListener("change", () => { const nn = nameIn.value.trim(); if (!nn || nn === bn || s.state[nn]) return; s.state[nn] = s.state[bn]; delete s.state[bn]; markDirty(); render(); });
  box.appendChild(h("div", { class: "sb-h" }, [nameIn, h("span", { class: "spacer", style: "flex:1" }), h("button", { class: "mini danger", onclick: () => { delete s.state[bn]; if (!Object.keys(s.state).length) delete s.state; markDirty(); render(); } }, ["✕"])]));
  const setBlk = (patch) => { s.state[bn] = Object.assign(blockDecl(s.state[bn]), patch); markDirty(); render(); };
  box.appendChild(fieldRow("width", numInput(d.width, v => setBlk({ width: coerce(v, "number") }))));
  box.appendChild(fieldRow("integ", selInput(d.integration, S.catalog.integrations, v => setBlk({ integration: v }))));
  box.appendChild(fieldRow("bound", selInput(d.boundary, S.catalog.state_boundaries, v => setBlk({ boundary: v }))));
  return box;
}

function inspField(name) {
  const f = S.spec.fields[name];
  const wrap = h("div", {});
  wrap.appendChild(h("div", { class: "insp-head" }, [h("span", { class: "nm" }, [name]), h("span", { class: "badge kind", style: `background:${S.catalog.field_color}` }, ["field"])]));
  const nameIn = h("input", { class: "in", type: "text", value: name });
  nameIn.addEventListener("change", () => renameField(name, nameIn.value));
  wrap.appendChild(fieldRow("name", nameIn));
  wrap.appendChild(fieldRow("kind", selInput(f.kind || "", S.catalog.field_kinds, v => { f.kind = v; markDirty(); render(); }, "—choose—")));
  wrap.appendChild(h("div", { class: "sec" }, ["config"]));
  for (const k of Object.keys(f)) { if (k === "kind") continue; wrap.appendChild(fieldRow(k, numInput(f[k], v => { f[k] = coerce(v, "number"); markDirty(); render(); }))); }
  wrap.appendChild(h("div", { class: "rowbtns" }, [
    h("button", { class: "mini", onclick: () => { const k = prompt("config key:"); if (k) { f[k] = 0; markDirty(); render(); } } }, ["+ key"]),
    h("button", { class: "mini danger", onclick: () => deleteField(name) }, ["delete field"]),
  ]));
  return wrap;
}

// =====================================================================
//  schedule rail
// =====================================================================
function renderSchedule() {
  const rail = $("#schedRail"); rail.textContent = "";
  if (!S.spec) return;
  const sched = S.spec.schedule || [];
  sched.forEach((tok, i) => {
    if (typeof tok !== "string") { rail.appendChild(h("div", { class: "sched-chip group" }, ["‹group›"])); return; }
    const def = opDef(tok);
    const selCls = (S.sel && S.sel.type === "op" && S.sel.id === tok) ? " sel" : "";
    const chip = h("div", { class: "sched-chip" + selCls, draggable: "true", style: `border-left-color:${def ? def.color : "#8b93a1"}`, "data-idx": i }, [
      h("span", { class: "idx" }, [String(i + 1)]),
      h("span", { class: "snm" }, [tok]),
      h("span", { class: "x", title: "remove from schedule" }, ["✕"]),
    ]);
    chip.querySelector(".snm").addEventListener("click", () => selectAndFocus("op", tok));
    chip.querySelector(".x").addEventListener("click", (e) => { e.stopPropagation(); sched.splice(i, 1); markDirty(); render(); });
    chip.addEventListener("dragstart", e => { e.dataTransfer.setData("text/plain", "sched:" + i); schedDragFrom = i; });
    chip.addEventListener("dragover", e => e.preventDefault());
    chip.addEventListener("drop", e => { e.preventDefault(); reorderSchedule(schedDragFrom, i); });
    rail.appendChild(chip);
  });
}
let schedDragFrom = null;
function reorderSchedule(from, to) {
  if (from == null || from === to) return;
  const sched = S.spec.schedule;
  const [item] = sched.splice(from, 1); sched.splice(to, 0, item);
  schedDragFrom = null; markDirty(); render();
}

// =====================================================================
//  mutations: add / delete / rename
// =====================================================================
function pickDefaultSet(def) {
  const sets = setNames(), fields = fieldNames();
  if (def && def.set && sets.includes(def.set)) return def.set;
  if (def && def.kind === "field" && fields.length) return fields[0];
  return sets[0] || fields[0] || "";
}
function newOp(name) {
  const def = opDef(name);
  const op = { op: name, at: pickDefaultSet(def) };
  if (def) {
    const impl = def.default_impl;
    if (Object.keys(def.implementations).length > 1) op.implementation = impl;
    for (const p of def.implementations[impl].params) if (p.default != null && !p.required) op[p.name] = p.default;
    if (def.kind === "exchange" && fieldNames().length) op.to = fieldNames()[0];
  }
  return op;
}
function addOp(name, x, y) {
  if (!S.spec) return toast("load or create a spec first", "err");
  if ((S.spec.operators || []).some(o => o.op === name))
    return toast(`'${name}' already in spec (a contract appears once)`, "err");
  const op = newOp(name);
  S.spec.operators = S.spec.operators || []; S.spec.operators.push(op);
  S.spec.schedule = S.spec.schedule || []; S.spec.schedule.push(name);
  if (x != null) S.layout.nodes[nodeId("op", name)] = { x, y };
  markDirty(); S.sel = { type: "op", id: name }; render();
}
function addOpAtCenter(name) {
  const r = $("#canvas").getBoundingClientRect();
  const p = screenToWorld(r.left + r.width / 2, r.top + r.height / 2);
  addOp(name, p.x - OPW / 2, p.y - 30);
}
function deleteOp(name) {
  S.spec.operators = (S.spec.operators || []).filter(o => o.op !== name);
  S.spec.schedule = (S.spec.schedule || []).filter(t => t !== name);
  delete S.layout.nodes[nodeId("op", name)];
  if (S.sel && S.sel.type === "op" && S.sel.id === name) S.sel = null;
  markDirty(); render();
}
function uniqueName(base, taken) { let n = base, i = 1; while (taken.includes(n)) n = base + (++i); return n; }
function addSet() {
  const name = uniqueName("set", setNames());
  S.spec.sets[name] = { n: 100 };
  markDirty(); S.sel = { type: "set", id: name }; render();
}
function deleteSet(name) {
  const used = (S.spec.operators || []).filter(o => o.at === name).map(o => o.op);
  if (used.length && !confirm(`${name} is used by: ${used.join(", ")}. Delete anyway?`)) return;
  delete S.spec.sets[name]; delete S.layout.nodes[nodeId("set", name)];
  if (S.sel && S.sel.type === "set" && S.sel.id === name) S.sel = null;
  markDirty(); render();
}
function renameSet(oldn, newn) {
  newn = (newn || "").trim();
  if (!newn || newn === oldn) return;
  if (setNames().includes(newn) || fieldNames().includes(newn)) return toast("name taken", "err");
  const sets = {}; for (const k of Object.keys(S.spec.sets)) sets[k === oldn ? newn : k] = S.spec.sets[k];
  S.spec.sets = sets;
  for (const o of (S.spec.operators || [])) if (o.at === oldn) o.at = newn;
  if (S.layout.nodes[nodeId("set", oldn)]) { S.layout.nodes[nodeId("set", newn)] = S.layout.nodes[nodeId("set", oldn)]; delete S.layout.nodes[nodeId("set", oldn)]; }
  S.sel = { type: "set", id: newn }; markDirty(); render();
}
function addField() {
  const name = uniqueName("field", fieldNames());
  S.spec.fields[name] = { kind: (S.catalog.field_kinds[0] || "grid") };
  markDirty(); S.sel = { type: "field", id: name }; render();
}
function deleteField(name) {
  const used = (S.spec.operators || []).filter(o => o.at === name || o.to === name || o.from === name).map(o => o.op);
  if (used.length && !confirm(`${name} is used by: ${used.join(", ")}. Delete anyway?`)) return;
  delete S.spec.fields[name]; delete S.layout.nodes[nodeId("field", name)];
  if (S.sel && S.sel.type === "field" && S.sel.id === name) S.sel = null;
  markDirty(); render();
}
function renameField(oldn, newn) {
  newn = (newn || "").trim();
  if (!newn || newn === oldn) return;
  if (setNames().includes(newn) || fieldNames().includes(newn)) return toast("name taken", "err");
  const f = {}; for (const k of Object.keys(S.spec.fields)) f[k === oldn ? newn : k] = S.spec.fields[k];
  S.spec.fields = f;
  for (const o of (S.spec.operators || [])) { if (o.to === oldn) o.to = newn; if (o.from === oldn) o.from = newn; if (o.at === oldn) o.at = newn; }
  if (S.layout.nodes[nodeId("field", oldn)]) { S.layout.nodes[nodeId("field", newn)] = S.layout.nodes[nodeId("field", oldn)]; delete S.layout.nodes[nodeId("field", oldn)]; }
  S.sel = { type: "field", id: newn }; markDirty(); render();
}
function addStateBlock(setName) {
  const s = S.spec.sets[setName]; s.state = s.state || {};
  const bn = uniqueName("chem", Object.keys(s.state));
  s.state[bn] = { width: 1, integration: "first_order", boundary: "free" };
  markDirty(); render();
}

// =====================================================================
//  selection + focus
// =====================================================================
function select(type, id) { S.sel = (type ? { type, id } : null); renderNodes(); renderInspector(); renderSchedule(); }
function selectAndFocus(type, id) {
  select(type, id);
  const g = S.geom[nodeId(type, id)]; if (!g) return;
  const r = $("#canvas").getBoundingClientRect();
  S.view.x = r.width / 2 - (g.x + g.w / 2) * S.view.k;
  S.view.y = r.height / 2 - (g.y + g.h / 2) * S.view.k;
  applyView();
}

// =====================================================================
//  pan / zoom / drag / wiring
// =====================================================================
function screenToWorld(px, py) {
  const r = $("#canvas").getBoundingClientRect();
  return { x: (px - r.left - S.view.x) / S.view.k, y: (py - r.top - S.view.y) / S.view.k };
}
let drag = null;   // {mode:'pan'|'node'|'wire', ...}

function onPointerDown(e) {
  const canvas = $("#canvas");
  const portEl = e.target.closest(".port");
  if (portEl) {
    const id = portEl.getAttribute("data-nodeid"), port = portEl.getAttribute("data-port");
    drag = { mode: "wire", from: { id, port } };
    const a = portAbs(id, port);
    $("#wire").appendChild(svg("path", { d: "" }));
    drag.wireA = a; canvas.setPointerCapture(e.pointerId); e.preventDefault(); return;
  }
  const nodeEl = e.target.closest(".node");
  if (nodeEl) {
    const id = nodeEl.getAttribute("data-nodeid"); const g = S.geom[id];
    const [type, ...rest] = id.split(":"); select(type, rest.join(":"));
    const w0 = screenToWorld(e.clientX, e.clientY);
    drag = { mode: "node", id, dx: w0.x - g.x, dy: w0.y - g.y, moved: false };
    canvas.setPointerCapture(e.pointerId); e.preventDefault(); return;
  }
  // background -> pan (and deselect)
  drag = { mode: "pan", x0: e.clientX, y0: e.clientY, vx: S.view.x, vy: S.view.y, moved: false };
  canvas.setPointerCapture(e.pointerId);
}
function onPointerMove(e) {
  if (!drag) return;
  if (drag.mode === "pan") {
    S.view.x = drag.vx + (e.clientX - drag.x0); S.view.y = drag.vy + (e.clientY - drag.y0);
    if (Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0) > 3) drag.moved = true;
    applyView();
  } else if (drag.mode === "node") {
    const w = screenToWorld(e.clientX, e.clientY);
    const g = S.geom[drag.id]; g.x = w.x - drag.dx; g.y = w.y - drag.dy; drag.moved = true;
    S.layout.nodes[drag.id] = { x: g.x, y: g.y };
    const nodeEl = $(`.node[data-nodeid="${cssEsc(drag.id)}"]`);
    if (nodeEl) nodeEl.setAttribute("transform", `translate(${g.x},${g.y})`);
    renderEdges();
  } else if (drag.mode === "wire") {
    const w = screenToWorld(e.clientX, e.clientY);
    const p = $("#wire path"); if (p) p.setAttribute("d", edgePath(drag.wireA, { x: w.x, y: w.y, face: -drag.wireA.face }));
  }
}
function onPointerUp(e) {
  if (!drag) return;
  if (drag.mode === "pan" && !drag.moved) select(null);
  if (drag.mode === "node" && drag.moved) persistLayout();
  if (drag.mode === "wire") {
    $("#wire").textContent = "";
    const tgtEl = document.elementFromPoint(e.clientX, e.clientY);
    const tnode = tgtEl && tgtEl.closest ? tgtEl.closest(".node") : null;
    if (tnode) applyWire(drag.from, tnode.getAttribute("data-nodeid"));
  }
  drag = null;
}
function applyWire(from, targetId) {
  const [ttype, ...rest] = targetId.split(":"); const tname = rest.join(":");
  const [ftype, ...frest] = from.id.split(":"); const fname = frest.join(":");
  if (ftype === "op") {
    const op = S.spec.operators.find(o => o.op === fname); if (!op) return;
    if (from.port === "act" && (ttype === "set" || ttype === "field")) op.at = tname;
    else if ((from.port === "from" || from.port === "to") && ttype === "field") op[from.port] = tname;
    else return;
    markDirty(); render();
  } else if ((ftype === "set" || ftype === "field") && ttype === "op") {
    const op = S.spec.operators.find(o => o.op === tname); if (!op) return;
    if (ftype === "set") op.at = fname; else op.from = fname;
    markDirty(); render();
  }
}
function onWheel(e) {
  e.preventDefault();
  const r = $("#canvas").getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const wx = (mx - S.view.x) / S.view.k, wy = (my - S.view.y) / S.view.k;
  const k = clamp(S.view.k * (e.deltaY < 0 ? 1.1 : 0.9), 0.25, 2.5);
  S.view.k = k; S.view.x = mx - wx * k; S.view.y = my - wy * k; applyView();
}
function cssEsc(s) { return (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/["\\]/g, "\\$&"); }
function fitView() {
  const ids = Object.keys(S.geom); if (!ids.length) return;
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const id of ids) { const g = S.geom[id]; x0 = Math.min(x0, g.x); y0 = Math.min(y0, g.y); x1 = Math.max(x1, g.x + g.w); y1 = Math.max(y1, g.y + g.h); }
  const r = $("#canvas").getBoundingClientRect(); const pad = 60;
  const k = clamp(Math.min((r.width - pad) / (x1 - x0), (r.height - pad) / (y1 - y0)), 0.25, 1.6);
  S.view.k = k; S.view.x = (r.width - (x1 + x0) * k) / 2; S.view.y = (r.height - (y1 + y0) * k) / 2; applyView();
}

// =====================================================================
//  persistence: validate / save / layout
// =====================================================================
async function runValidate() {
  if (!S.spec) return;
  const res = await apiPost("/api/validate", { spec: S.spec });
  const pill = $("#validPill");
  if (res.valid) { pill.className = "pill ok"; pill.textContent = "valid"; $("#statusMsg").textContent = ""; }
  else { pill.className = "pill err"; pill.textContent = "invalid"; $("#statusMsg").textContent = res.error || ""; }
  return res.valid;
}
async function persistLayout() {
  if (!S.specPath) return;
  S.layout.view = S.view;
  apiPost("/api/layout", { path: S.specPath, layout: S.layout });
}
async function doSave(force = false) {
  if (!S.spec) return;
  if (!S.specPath) {
    const suggested = (S.repoRoot || "") + "/prototype/" + (S.spec.general.name || "untitled") + "/spec.yaml";
    const p = prompt("save spec to (absolute path under the repo):", suggested);
    if (!p) return; S.specPath = p;
  }
  S.layout.view = S.view;
  const res = await apiPost("/api/save", { path: S.specPath, spec: S.spec, layout: S.layout, force });
  if (res.saved) {
    S.dirty = false; S.forceSaveArmed = false; $("#btnSave").textContent = "SAVE";
    S.specRel = res.rel || S.specRel; S.specPath = res.path || S.specPath;
    $("#specPath").textContent = S.specRel || S.specPath;
    history.replaceState(null, "", "?spec=" + encodeURIComponent(S.specPath));
    toast(res.valid ? "saved ✓" : "saved (invalid) ⚠", res.valid ? "ok" : "err");
    runValidate();
  } else {
    $("#statusMsg").textContent = res.error || "invalid";
    if (confirm("Spec is invalid:\n\n" + (res.error || "") + "\n\nSave anyway?")) doSave(true);
  }
}

// =====================================================================
//  load
// =====================================================================
function autoLayout() { S.layout = { nodes: {}, view: { x: 60, y: 40, k: 1 } }; }
async function loadSpec(path) {
  const res = await apiGet("/api/spec?path=" + encodeURIComponent(path));
  if (res.error) { toast(res.error, "err"); return; }
  S.spec = res.spec || {};
  S.spec.general = S.spec.general || {}; S.spec.sets = S.spec.sets || {};
  S.spec.fields = S.spec.fields || {}; S.spec.operators = S.spec.operators || []; S.spec.schedule = S.spec.schedule || [];
  S.specPath = res.path; S.specRel = res.rel; S.media = res.media || null;
  S.layout = res.layout && res.layout.nodes ? res.layout : { nodes: {}, view: { x: 60, y: 40, k: 1 } };
  S.view = Object.assign({ x: 60, y: 40, k: 1 }, S.layout.view || {});
  S.sel = null; S.dirty = false; $("#btnSave").textContent = "SAVE";
  $("#specPath").textContent = S.specRel || S.specPath;
  history.replaceState(null, "", "?spec=" + encodeURIComponent(S.specPath));
  render(); renderMedia();
  if (!S.layout.nodes || !Object.keys(S.layout.nodes).length) fitView();
  runValidate();
}
function newSpec() {
  S.spec = { general: { name: "untitled", seed: 0, n_frames: 200, dt: 0.05, boundary: "wall", dim: 2, world: [1.0, 1.0] },
             sets: { cell: { n: 100 } }, fields: {}, operators: [], schedule: [] };
  S.specPath = null; S.specRel = null; S.media = null; autoLayout(); S.view = { x: 60, y: 40, k: 1 };
  S.sel = { type: "general" }; S.dirty = true; $("#btnSave").textContent = "SAVE *";
  $("#specPath").textContent = "untitled (unsaved)";
  render(); renderMedia(); runValidate();
}

// ---- load modal ----
async function openLoadModal() {
  const res = await apiGet("/api/specs");
  S.repoRoot = res.repo_root || S.repoRoot;
  const list = $("#specList"); list.textContent = "";
  const q = ($("#specFilter").value || "").toLowerCase();
  const matches = res.specs.filter(sp => !q || sp.rel.toLowerCase().includes(q));
  const CAP = 400;
  for (const sp of matches.slice(0, CAP)) {
    const parts = sp.rel.split("/"); const file = parts.pop();
    const item = h("div", { class: "spec-item" }, [
      h("span", { class: "dirp" }, [parts.join("/") + "/"]),
      h("span", { class: "filep" }, [file]),
    ]);
    item.addEventListener("click", () => { closeModal(); loadSpec(sp.path); });
    list.appendChild(item);
  }
  if (matches.length > CAP)
    list.appendChild(h("div", { class: "spec-item dim" }, [`… ${matches.length - CAP} more — type to filter (${matches.length} match)`]));
  $("#modal").classList.remove("hidden");
}
function closeModal() { $("#modal").classList.add("hidden"); }

// ---- mp4 viewer (bottom-right) ----
function renderMedia() {
  const card = $("#media"), vid = $("#mediaVideo");
  if (!S.media) { card.classList.add("hidden"); vid.removeAttribute("src"); try { vid.load(); } catch (e) {} return; }
  $("#mediaName").textContent = S.media.name || "movie.mp4";
  if (S.media.poster) vid.setAttribute("poster", S.media.poster); else vid.removeAttribute("poster");
  vid.setAttribute("src", S.media.video);
  try { vid.load(); } catch (e) {}
  card.classList.remove("hidden", "collapsed");
}

// =====================================================================
//  wire up
// =====================================================================
async function init() {
  indexCatalog(await apiGet("/api/catalog"));
  try { const sp = await apiGet("/api/specs"); S.repoRoot = sp.repo_root || ""; } catch (e) {}
  renderPalette();

  $("#btnNew").addEventListener("click", newSpec);
  $("#btnLoad").addEventListener("click", openLoadModal);
  $("#btnValidate").addEventListener("click", () => runValidate().then(v => toast(v ? "valid ✓" : "invalid ✗", v ? "ok" : "err")));
  $("#btnSave").addEventListener("click", () => doSave(false));
  $("#btnFit").addEventListener("click", fitView);
  $("#tglSchedule").addEventListener("change", e => { S.showSchedule = e.target.checked; renderEdges(); });
  $("#palFilter").addEventListener("input", renderPalette);
  $("#modalClose").addEventListener("click", closeModal);
  $("#specFilter").addEventListener("input", openLoadModal);
  $("#modal").addEventListener("click", e => { if (e.target.id === "modal") closeModal(); });
  $("#mediaClose").addEventListener("click", () => { S.media = null; renderMedia(); });
  $("#mediaCollapse").addEventListener("click", () => {
    const c = $("#media").classList.toggle("collapsed");
    $("#mediaCollapse").textContent = c ? "+" : "–";
  });

  const canvas = $("#canvas");
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  const wrap = $("#canvas-wrap");
  wrap.addEventListener("dragover", e => e.preventDefault());
  wrap.addEventListener("drop", e => {
    e.preventDefault(); const name = e.dataTransfer.getData("text/plain");
    if (name && S.opByName[name]) { const p = screenToWorld(e.clientX, e.clientY); addOp(name, p.x - OPW / 2, p.y - 30); }
  });
  document.addEventListener("keydown", e => {
    if (e.target.matches("input,select,textarea")) return;
    if ((e.key === "Delete" || e.key === "Backspace") && S.sel) {
      if (S.sel.type === "op") deleteOp(S.sel.id);
      else if (S.sel.type === "set") deleteSet(S.sel.id);
      else if (S.sel.type === "field") deleteField(S.sel.id);
    }
    if (e.key === "f") fitView();
    if (e.key === "Escape") { closeModal(); select(null); }
  });

  const params = new URLSearchParams(location.search);
  const sp = params.get("spec");
  if (sp) loadSpec(sp); else openLoadModal();
}
init();
