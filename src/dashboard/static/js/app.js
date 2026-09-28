/* X-UBA Identity Risk Intelligence dashboard - MCA Major Project.
 * Every number comes from the Flask JSON API (src/dashboard/app.py). */
"use strict";

const RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
const RISK_COLORS = { CRITICAL: "#f43f5e", HIGH: "#f59e0b", MEDIUM: "#3b82f6", LOW: "#10d97a" };
const QUAD_COLORS = { QUIET_RISK: "#a855f7", CONSENSUS_RISK: "#f43f5e", NOISY_BENIGN: "#f59e0b", CONSENSUS_BENIGN: "#10d97a" };
const SENS_COLORS = { critical: "#f43f5e", high: "#f59e0b", medium: "#3b82f6", low: "#10d97a" };
const STRAT_COLORS = ["#5b6478", "#a855f7", "#6366f1"];

/* ── helpers ── */
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => {
  if (v === null || v === undefined || Number.isNaN(+v)) return "—";
  const s = (+v).toFixed(d);
  return /^-0(\.0+)?$/.test(s) ? s.slice(1) : s;   // never show "-0.0"
};
const int = (v) => (v === null || v === undefined) ? "—" : Math.round(+v).toLocaleString();
const pct = (v, d = 1) => (v === null || v === undefined) ? "—" : (+v * 100).toFixed(d) + "%";
const badge = (v) => v ? `<span class="badge ${esc(v)}">${esc(String(v).replace(/_/g, " "))}</span>` : "—";
const spinner = (msg = "Loading...") => `<div class="loading"><div class="spinner"></div>${esc(msg)}</div>`;
const nice = (s) => String(s).replace(/_/g, " ");

async function api(url, opts) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}) });

function fail(el, err) { el.innerHTML = `<div class="loading error">${esc(err.message || err)}</div>`; }

function barRows(entries, { color = "linear-gradient(90deg,#6366f1,#a855f7)", digits = 0, max = null, label = nice } = {}) {
  if (!entries.length) return `<div class="empty">None</div>`;
  const m = max ?? Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);
  return entries.map(([k, v]) => `<div class="bar-row"><div class="bar-label" title="${esc(k)}">${esc(label(k))}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, Math.abs(v) / m * 100)}%;background:${
      typeof color === "function" ? color(k, v) : color}"></div></div>
    <div class="bar-value">${digits ? fmt(v, digits) : int(v)}</div></div>`).join("");
}

/* ── charts ── */
const charts = {};
if (window.Chart) {
  Chart.defaults.font.family = "-apple-system, 'Segoe UI', system-ui, sans-serif";
  Chart.defaults.color = "#8a93a6";
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
}
const GRID = { color: "rgba(255,255,255,.05)" };
function chart(id, config) {
  const el = $(id);
  if (!el) return;
  if (!window.Chart) {
    el.parentElement.innerHTML = `<div class="loading error">Chart library missing: src/dashboard/static/js/chart.umd.min.js</div>`;
    return;
  }
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(el, config);
}
const axes = (extra = {}) => ({ x: { grid: GRID, ...(extra.x || {}) }, y: { grid: GRID, ...(extra.y || {}) } });

/* ── tabs ── */
const loaded = {};
function positionIndicator(tab) {
  const ind = $("tab-indicator");
  ind.style.left = tab.offsetLeft + "px";
  ind.style.width = tab.offsetWidth + "px";
}
function showPage(page) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.page === page));
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + page));
  positionIndicator(document.querySelector(`.tab[data-page="${page}"]`));
  if (!loaded[page] && LOADERS[page]) { loaded[page] = true; LOADERS[page](); }
  try { history.replaceState(null, "", "#" + page); } catch (e) { /* file:// etc. */ }
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showPage(t.dataset.page)));
window.addEventListener("resize", () => positionIndicator(document.querySelector(".tab.active")));

function animateCount(el, target, decimals = 0, suffix = "") {
  const t0 = performance.now(), dur = 700;
  const step = (t) => {
    const p = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
    el.textContent = (decimals ? (target * e).toFixed(decimals) : Math.round(target * e).toLocaleString()) + suffix;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
function kpi(cls, label, value, sub, decimals = 0, suffix = "") {
  return { html: `<div class="card kpi ${cls}"><div class="label">${esc(label)}</div>
    <div class="value" data-v="${+value || 0}" data-d="${decimals}" data-s="${esc(suffix)}">0</div><div class="delta">${sub}</div></div>` };
}
function renderKpis(el, items) {
  el.innerHTML = items.map((k) => k.html).join("");
  el.querySelectorAll(".value[data-v]").forEach((v) => animateCount(v, +v.dataset.v, +v.dataset.d, v.dataset.s));
}

/* ═════════════ OVERVIEW ═════════════ */
async function loadOverview() {
  try {
    const s = await api("/api/summary");
    const n = s.total_identities;
    $("live-text").textContent = `${n.toLocaleString()} identities · models loaded`;
    renderKpis($("kpi-row"), [
      kpi("blue", "Identities monitored", n, "480k telemetry events"),
      kpi("red", "Predicted critical", s.risk_distribution.CRITICAL, `${pct(s.risk_distribution.CRITICAL / n)} of the estate`),
      kpi("purple", "Quiet risk", s.quiet_risk_count, "behaviourally quiet, structurally exposed"),
      kpi("orange", "SoD violators", s.sod_violation_identities, `${pct(s.sod_violation_identities / n)} of identities · ${int(s.mfa_gap_count)} without MFA`),
      kpi("green", "Avg compliance", s.avg_compliance, `${int(s.compliance_gap_identities)} identities with a gap`, 1),
      kpi("blue", "Early warnings", s.early_warnings, `avg blast radius ${fmt(s.avg_blast_radius)}/100`),
    ]);
    const labels = RISK_ORDER.filter((k) => k in s.risk_distribution);
    chart("chart-risk-dist", { type: "doughnut",
      data: { labels, datasets: [{ data: labels.map((k) => s.risk_distribution[k]),
        backgroundColor: labels.map((k) => RISK_COLORS[k]), borderColor: "#10141d", borderWidth: 3 }] },
      options: { cutout: "66%", plugins: { legend: { position: "bottom" } } } });
    const tl = Object.keys(s.threat_distribution);
    chart("chart-threat-dist", { type: "bar",
      data: { labels: tl.map(nice), datasets: [{ data: tl.map((k) => s.threat_distribution[k]),
        backgroundColor: "#6366f1", borderRadius: 6 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: axes({ y: { grid: { display: false } } }) } });
    $("cluster-list").innerHTML = s.clusters.map((c) => `
      <div style="margin-bottom:12px"><div style="display:flex;justify-content:space-between;gap:8px;font-size:12.5px">
        <span>${esc(c.cluster_name)}</span><span class="mono">${int(c.n_identities)}</span></div>
      <div class="bar-track" style="margin-top:5px"><div class="bar-fill" style="width:${c.pct_true_anomaly}%;
        background:linear-gradient(90deg,#f59e0b,#f43f5e)"></div></div>
      <div class="note" style="margin-top:3px">${fmt(c.pct_true_anomaly)}% true anomalies · ${fmt(c.pct_admin)}% admin ·
        ${fmt(c.pct_mfa_enabled)}% MFA</div></div>`).join("");
    $("top-targets").innerHTML = s.top_targets.map((r) => `
      <tr class="click" data-id="${esc(r.identity_id)}"><td>${r.attack_rank}</td><td class="mono">${esc(r.identity_id)}</td>
      <td>${esc(r.name)}</td><td>${esc(r.department)}</td><td>${esc(r.privilege_level)}</td>
      <td class="n">${fmt(r.xgb_risk_index)}</td><td>${badge(r.xgb_risk_level_pred)}</td>
      <td class="n">${fmt(r.blast_radius_score)}</td><td class="n">${fmt(r.attack_likelihood, 3)}</td>
      <td class="n">${fmt(r.expected_attack_risk, 2)}</td></tr>`).join("");
  } catch (e) { fail($("kpi-row"), e); }
}

/* ═════════════ EXPLORER ═════════════ */
let page = 1, pages = 1;
async function loadDepartments() {
  const depts = await api("/api/departments");
  const sel = $("f-department");
  depts.forEach((d) => sel.insertAdjacentHTML("beforeend", `<option value="${esc(d)}">${esc(d)}</option>`));
}
async function loadIdentities(p = 1) {
  page = p;
  const tb = $("explorer-tbody");
  tb.innerHTML = `<tr><td colspan="13">${spinner()}</td></tr>`;
  const params = new URLSearchParams({ search: $("f-search").value, department: $("f-department").value,
    risk_level: $("f-risk").value, quadrant: $("f-quadrant").value, sort: $("f-sort").value, page, page_size: 50 });
  try {
    const d = await api("/api/identities?" + params);
    pages = d.pages;
    $("explorer-count").textContent = `${d.total.toLocaleString()} matching`;
    $("page-indicator").textContent = `Page ${d.page} of ${d.pages}`;
    $("prev-page").disabled = d.page <= 1;
    $("next-page").disabled = d.page >= d.pages;
    tb.innerHTML = d.results.length ? d.results.map((r) => `
      <tr class="click" data-id="${esc(r.identity_id)}"><td class="mono">${esc(r.identity_id)}</td><td>${esc(r.name)}</td>
      <td>${esc(r.department)}</td><td>${esc(r.privilege_level)}</td><td>${r.mfa_enabled ? "yes" : "<b style='color:#fda4af'>NO</b>"}</td>
      <td class="n">${fmt(r.xgb_risk_index)}</td><td>${badge(r.xgb_risk_level_pred)}</td><td>${esc(nice(r.xgb_threat_pred))}</td>
      <td class="n">${fmt(r.fused_risk_score)}</td><td class="n">${fmt(r.blast_radius_score)}</td><td>${badge(r.quadrant)}</td>
      <td class="n">${r.sod_violation_count}</td><td class="n">${fmt(r.compliance_score, 0)}</td></tr>`).join("")
      : `<tr><td colspan="13" class="empty">No identities match these filters.</td></tr>`;
  } catch (e) { tb.innerHTML = `<tr><td colspan="13" class="empty error">${esc(e.message)}</td></tr>`; }
}
$("explorer-form").addEventListener("submit", (e) => { e.preventDefault(); loadIdentities(1); });
$("prev-page").addEventListener("click", () => page > 1 && loadIdentities(page - 1));
$("next-page").addEventListener("click", () => page < pages && loadIdentities(page + 1));

/* ═════════════ IDENTITY DEEP-DIVE ═════════════ */
let current = null;
document.addEventListener("click", (e) => {
  const row = e.target.closest("tr.click[data-id]");
  if (row) openIdentity(row.dataset.id);
});
$("modal-close").addEventListener("click", closeModal);
$("modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
function closeModal() { $("modal").classList.remove("show"); }

function probBars(probs) {
  return RISK_ORDER.filter((k) => k in probs).map((k) => `<div class="bar-row"><div class="bar-label">${k}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${probs[k] * 100}%;background:${RISK_COLORS[k]}"></div></div>
    <div class="bar-value">${pct(probs[k], 0)}</div></div>`).join("");
}
function chainHtml(a) {
  const st = [["Initial access", a.p_initial_access], ["Privilege escalation", a.p_privilege_escalation],
    ["Lateral movement", a.p_lateral_movement], ["Data impact", a.p_data_impact]];
  return `<div class="chain">${st.map(([t, p], i) => `<div class="step"><div class="t">${i + 1}. ${t}</div>
    <div class="p">${pct(p, 0)}</div></div>`).join("")}
    <div class="step total"><div class="t">End-to-end likelihood</div><div class="p">${pct(a.attack_likelihood, 1)}</div></div></div>`;
}
function beforeAfter(b, a, title) {
  const d = a.risk_index - b.risk_index;
  const ear = a.attack.expected_attack_risk - b.attack.expected_attack_risk;
  return `<div class="ba">
    <div class="col"><div class="t">Before</div><div class="big" style="color:${RISK_COLORS[b.predicted_class]}">${fmt(b.risk_index, 0)}</div>${badge(b.predicted_class)}</div>
    <div class="arrow">&rarr;</div>
    <div class="col"><div class="t">After</div><div class="big" style="color:${RISK_COLORS[a.predicted_class]}">${fmt(a.risk_index, 0)}</div>${badge(a.predicted_class)}</div>
    <div class="col"><div class="t">Risk change</div><div class="big" style="color:${d < 0 ? "#6ee7b7" : "#8a93a6"}">${d > 0 ? "+" : ""}${fmt(d, 1)}</div></div>
    <div class="col"><div class="t">Attack risk</div><div class="big" style="font-size:18px">${fmt(b.attack.expected_attack_risk, 2)} &rarr; ${fmt(a.attack.expected_attack_risk, 2)}</div>
      <div class="note" style="margin:0">${ear < 0 ? "reduced" : "unchanged"} · blast ${fmt(b.attack.blast_radius_score, 0)} &rarr; ${fmt(a.attack.blast_radius_score, 0)}</div></div>
  </div>${title ? `<div class="note">${title}</div>` : ""}`;
}

async function openIdentity(id) {
  $("modal").classList.add("show");
  $("modal-title").textContent = id;
  $("modal-sub").textContent = "";
  $("modal-body").innerHTML = spinner("Loading identity, computing live SHAP...");
  let d;
  try { d = await api("/api/identity/" + encodeURIComponent(id)); }
  catch (e) { return fail($("modal-body"), e); }
  current = d;
  $("modal-title").innerHTML = `${esc(d.name)} <span class="mono" style="color:#8a93a6">${esc(d.identity_id)}</span> ${badge(d.xgb_risk_level_pred)} ${badge(d.quadrant)}`;
  $("modal-sub").textContent = `${d.department} · ${d.job_title} · ${d.privilege_level} · MFA ${d.mfa_enabled ? "on" : "OFF"} · inactive ${d.inactive_days} days` +
    (d.is_contractor ? (d.is_contractor_expired ? " · contractor (EXPIRED)" : " · contractor") : "") + (d.is_service ? " · service account" : "");
  const smax = Math.max(...d.shap_live.map((f) => Math.abs(f.shap)), 1e-9);
  const w = d.whatif_defaults;
  const month = d.monthly_signal;
  $("modal-body").innerHTML = `
  <div class="grid grid-6" style="margin-bottom:14px">
    ${[["Risk (cross-validated)", fmt(d.xgb_risk_index), "0-100 risk index"], ["Fused score", fmt(d.fused_risk_score), "4-signal fusion"],
      ["Blast radius", fmt(d.blast_radius_score), `${d.reachable_resources} resources · ${d.reachable_critical} critical`],
      ["Attack rank", `#${int(d.attack_rank)}`, `expected risk ${fmt(d.expected_attack_risk, 2)}`],
      ["Compliance", fmt(d.compliance_score, 0), `${d.gap_count} control gap(s)`],
      ["SoD violations", d.sod_violation_count, `severity ${d.sod_severity_score}`]]
      .map(([t, v, s]) => `<div class="panel-mini tile"><div class="panel-mini-title">${t}</div><div class="v">${v}</div><div class="s">${esc(s)}</div></div>`).join("")}
  </div>
  <div class="grid grid-2">
    <div class="panel-mini"><div class="panel-mini-title"><span>WHY &mdash; live SHAP (deployed model, class ${esc(d.live.predicted_class)})</span></div>
      ${d.shap_live.map((f) => `<div class="bar-row"><div class="bar-label" title="${esc(f.feature)}">${esc(f.feature)} = ${esc(f.value)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.abs(f.shap) / smax * 100}%;background:${f.shap > 0 ? "#f43f5e" : "#10d97a"}"></div></div>
        <div class="bar-value" style="color:${f.shap > 0 ? "#fda4af" : "#6ee7b7"}">${f.shap > 0 ? "+" : ""}${fmt(f.shap, 3)}</div></div>`).join("")}
      <div class="note">Red pushes towards <b>${esc(d.live.predicted_class)}</b>, green away from it. Predicted threat: <b>${esc(nice(d.xgb_threat_pred))}</b>.</div>
    </div>
    <div class="panel-mini"><div class="panel-mini-title"><span>Class probabilities (deployed model)</span></div>${probBars(d.live.probabilities)}
      <div class="panel-mini-title" style="margin-top:16px"><span>WHO / WHEN</span></div>
      <div class="chips" style="margin-top:0"><span class="chip">Cluster: <b>${esc(d.cluster_name)}</b></span>
        <span class="chip">Trajectory: <b>${esc(d.trajectory)}</b></span>${d.early_warning ? `<span class="chip" style="color:#fda4af">EARLY WARNING</span>` : ""}</div>
      <div class="chart-box" style="height:110px;margin-top:10px"><canvas id="chart-monthly"></canvas></div></div>
  </div>
  <div class="panel-mini mt"><div class="panel-mini-title"><span>WHAT &mdash; kill-chain simulation</span><span>IMPACT</span></div>
    ${chainHtml(d)}
    <div class="note">Attack path: <code>${esc(d.attack_path || "n/a")}</code> · reaches ${d.reachable_resources} resources on ${d.reachable_systems} systems
      (${d.reachable_critical} critical, ${d.reachable_high} high) · <a href="#" id="to-graph">open in privilege graph &rarr;</a></div></div>
  <div class="grid grid-2 mt">
    <div class="panel-mini"><div class="panel-mini-title"><span>ACTION &mdash; what-if simulator (live model)</span></div>
      <div class="form-row"><label><span>MFA</span><span id="wi-mfa-v"></span></label>
        <select id="wi-mfa"><option value="1">Enabled</option><option value="0">Disabled</option></select></div>
      <div class="form-row"><label><span>Privilege level</span><span id="wi-priv-v"></span></label>
        <select id="wi-priv"><option>user</option><option>power-user</option><option>admin</option></select></div>
      <div class="form-row"><label><span>Days inactive</span><span id="wi-inactive-v"></span></label><input type="range" id="wi-inactive" min="0" max="400" step="5"></div>
      <div class="form-row"><label><span>Number of systems</span><span id="wi-systems-v"></span></label><input type="range" id="wi-systems" min="1" max="15" step="1"></div>
      <div class="form-row"><label><span>Unapproved privilege changes</span><span id="wi-unapp-v"></span></label><input type="range" id="wi-unapp" min="0" max="10" step="1"></div>
      <div style="display:flex;gap:8px"><button id="wi-run" style="flex:1">Recompute live prediction</button><button class="ghost" id="wi-reset">Reset</button></div>
      <div id="wi-result"></div></div>
    <div class="panel-mini"><div class="panel-mini-title"><span>Counterfactual actions (precomputed)</span></div>
      ${d.counterfactual.length ? `<table><thead><tr><th>Action</th><th class="n">Risk change</th><th></th></tr></thead><tbody>
        ${d.counterfactual.map((c) => `<tr><td title="${esc(c.description)}">${esc(nice(c.action))}<div class="note" style="margin:0">${esc(c.description)}</div></td>
          <td class="n" style="color:${c.delta < 0 ? "#6ee7b7" : "#8a93a6"}">${c.delta > 0 ? "+" : ""}${fmt(c.delta)}</td>
          <td><button class="ghost" data-act="${esc(c.action)}" style="padding:5px 10px">Apply</button></td></tr>`).join("")}</tbody></table>`
        : `<div class="note">Not predicted HIGH/CRITICAL, so no precomputed plan. One-click remediation still picks the best action live.</div>`}
      <button class="go" id="remediate" style="width:100%;margin-top:12px">&#9889; One-click remediate (best action)</button>
      <div id="rem-result"></div></div>
  </div>
  <div class="grid grid-2 mt">
    <div class="panel-mini"><div class="panel-mini-title"><span>Separation-of-duties violations</span></div>
      ${d.sod_violations.length ? d.sod_violations.map((v) => `<div style="margin-bottom:10px;font-size:12.5px">${badge(v.severity === "CRITICAL" ? "CRITICAL" : "HIGH")}
        <b style="margin-left:4px">${esc(nice(v.rule))}</b><div class="note" style="margin-top:3px">${esc(v.evidence)}</div></div>`).join("")
        : `<div class="note">No violations.</div>`}</div>
    <div class="panel-mini"><div class="panel-mini-title"><span>NIST / GDPR controls</span><span>score ${fmt(d.compliance_score, 0)}/100</span></div>
      ${d.controls.map((c) => `<div class="ctrl"><span class="dot" style="background:${c.failed ? "#f43f5e" : "#10d97a"}"></span>${esc(nice(c.control))}
        <span style="margin-left:auto;color:${c.failed ? "#fda4af" : "#6ee7b7"}">${c.failed ? "GAP" : "OK"}</span></div>`).join("")}</div>
  </div>
  <div class="note mt">Ground truth (synthetic, used only for validation): ${esc(d.true_risk_level)} · ${esc(nice(d.true_threat_type))} ·
    anomaly ${d.true_is_anomaly ? "yes" : "no"} · ${esc(d.split)} set · JSON: <a href="/api/identity/${esc(d.identity_id)}" target="_blank">/api/identity/${esc(d.identity_id)}</a></div>`;

  if (window.Chart) chart("chart-monthly", { type: "line",
    data: { labels: month.map((_, i) => "W" + (i + 1)), datasets: [{ data: month, borderColor: "#818cf8", backgroundColor: "rgba(99,102,241,.15)",
      fill: true, tension: .35, pointRadius: 2 }] },
    options: { plugins: { legend: { display: false } }, scales: axes({ x: { ticks: { font: { size: 9 } } }, y: { ticks: { font: { size: 9 } } } }) } });

  const resetWhatIf = () => {
    $("wi-mfa").value = String(w.mfa_enabled); $("wi-priv").value = w.privilege_level;
    $("wi-inactive").value = w.inactive_days; $("wi-systems").value = w.n_systems; $("wi-unapp").value = Math.min(w.unapproved_changes, 10);
    syncLabels(); $("wi-result").innerHTML = "";
  };
  const syncLabels = () => {
    $("wi-mfa-v").textContent = $("wi-mfa").value === "1" ? "on" : "off";
    $("wi-priv-v").textContent = $("wi-priv").value;
    $("wi-inactive-v").textContent = $("wi-inactive").value;
    $("wi-systems-v").textContent = $("wi-systems").value;
    $("wi-unapp-v").textContent = $("wi-unapp").value;
  };
  ["wi-mfa", "wi-priv", "wi-inactive", "wi-systems", "wi-unapp"].forEach((k) => $(k).addEventListener("input", syncLabels));
  resetWhatIf();
  $("wi-reset").addEventListener("click", resetWhatIf);
  $("wi-run").addEventListener("click", runWhatIf);
  $("remediate").addEventListener("click", () => remediate(null));
  document.querySelectorAll("#modal-body button[data-act]").forEach((b) => b.addEventListener("click", () => remediate(b.dataset.act)));
  $("to-graph").addEventListener("click", (e) => { e.preventDefault(); closeModal(); showPage("graph"); $("graph-id").value = d.identity_id; loadEgo(d.identity_id); });
}

async function runWhatIf() {
  const out = $("wi-result");
  out.innerHTML = spinner("Re-scoring with the trained model...");
  try {
    const r = await post("/api/whatif/" + encodeURIComponent(current.identity_id), {
      mfa_enabled: +$("wi-mfa").value, privilege_level: $("wi-priv").value, inactive_days: +$("wi-inactive").value,
      n_systems: +$("wi-systems").value, unapproved_changes: +$("wi-unapp").value });
    out.innerHTML = beforeAfter(r.before, r.after, esc(r.note)) +
      `<div class="panel-mini-title" style="margin-top:12px"><span>New class probabilities</span></div>${probBars(r.after.probabilities)}`;
  } catch (e) { fail(out, e); }
}

async function remediate(action) {
  const out = $("rem-result");
  out.innerHTML = spinner("Applying action and re-scoring...");
  try {
    const r = await post("/api/remediate/" + encodeURIComponent(current.identity_id), action ? { action } : {});
    out.innerHTML = beforeAfter(r.before, r.after,
      `Applied <b>${esc(nice(r.action_applied))}</b> (${esc(r.description)}): same transform as the counterfactual engine, re-scored live;
       blast radius re-computed on the privilege graph where the action changes reach.`);
  } catch (e) { fail(out, e); }
}

/* ═════════════ PRIVILEGE GRAPH ═════════════ */
function forceGraph(el, nodes, edges, { color, radius, width: wfn, label = (d) => d.label, height = 500, dist = 70, charge = -160, ring = null }) {
  el.innerHTML = "";
  if (!window.d3) { el.innerHTML = `<div class="loading error">D3 missing: src/dashboard/static/js/d3.min.js</div>`; return; }
  const width = el.clientWidth || 700;
  const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
  const g = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (e) => g.attr("transform", e.transform)));
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id((d) => d.id).distance(dist).strength(ring ? 0.02 : 0.25))
    .force("charge", d3.forceManyBody().strength(charge))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide((d) => radius(d) + 6));
  if (ring) {  // two concentric rings: ring(d) returns the target radius for a node
    const R = Math.min(width, height) / 2 - 40;
    sim.force("radial", d3.forceRadial((d) => ring(d) * R, width / 2, height / 2).strength(1));
  }
  const link = g.append("g").selectAll("line").data(edges).join("line")
    .attr("stroke", (d) => d.kind === "admin" ? "#a855f7" : "#2d374a").attr("stroke-opacity", 0.75)
    .attr("stroke-width", wfn || 1.2).attr("stroke-dasharray", (d) => d.kind === "admin" ? "4 3" : null);
  const node = g.append("g").selectAll("circle").data(nodes).join("circle")
    .attr("r", radius).attr("fill", color).attr("stroke", "#080a10").attr("stroke-width", 2).style("cursor", "grab")
    .call(d3.drag()
      .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
  node.append("title").text((d) => d.title || d.label);
  const text = g.append("g").selectAll("text").data(nodes.filter((d) => label(d))).join("text")
    .text(label).attr("font-size", 10).attr("fill", "#9aa3b8").attr("dx", (d) => radius(d) + 4).attr("dy", 4)
    .style("pointer-events", "none");
  sim.on("tick", () => {
    link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    text.attr("x", (d) => d.x).attr("y", (d) => d.y);
  });
}

async function loadGraph() {
  const el = $("graph-overview");
  el.innerHTML = spinner("Building graph...");
  try {
    const g = await api("/api/graph/overview");
    const maxW = Math.max(...g.edges.map((e) => e.weight));
    const risks = g.nodes.filter((n) => n.type === "department").map((n) => n.risk);
    const lo = Math.min(...risks), hi = Math.max(...risks);
    g.nodes.forEach((n) => { n.title = n.type === "department" ? `${n.label}: avg risk ${n.risk}` : `${n.label} (${n.sensitivity})`; });
    forceGraph(el, g.nodes, g.edges, {
      color: (d) => d.type === "department" ? d3.interpolateRgb("#818cf8", "#f43f5e")((d.risk - lo) / ((hi - lo) || 1)) : SENS_COLORS[d.sensitivity] || "#10d97a",
      radius: (d) => d.type === "department" ? 13 : 8,
      width: (e) => 0.5 + 4 * e.weight / maxW, dist: 150, charge: -300,
      ring: (d) => d.type === "department" ? 0.55 : 1 });
  } catch (e) { fail(el, e); }
}

async function loadEgo(id) {
  const el = $("graph-identity");
  el.innerHTML = spinner();
  $("graph-chips").innerHTML = "";
  try {
    const g = await api("/api/graph/identity/" + encodeURIComponent(id.trim()));
    g.nodes.forEach((n) => { n.title = n.type === "resource" ? `${n.label} (${n.sensitivity})` : n.label; });
    forceGraph(el, g.nodes, g.edges, {
      color: (d) => d.type === "identity" ? "#f43f5e" : d.type === "system" ? (d.admin ? "#a855f7" : "#10d97a") : SENS_COLORS[d.sensitivity],
      radius: (d) => d.type === "identity" ? 14 : d.type === "system" ? 10 : 5,
      label: (d) => d.type === "resource" ? "" : d.label, dist: 55 });
    const s = g.summary;
    $("graph-chips").innerHTML = [["Reachable resources", s.reachable_resources], ["Critical", s.reachable_critical],
      ["High", s.reachable_high], ["Systems", s.reachable_systems], ["Admin systems", s.admin_systems],
      ["Direct accesses", g.direct_resources], ["Shown", g.shown_resources]]
      .map(([k, v]) => `<span class="chip">${k}: <b>${int(v)}</b></span>`).join("") +
      `<span class="chip">Attack path: <b>${esc(s.attack_path || "n/a")}</b></span>`;
  } catch (e) { fail(el, e); }
}
$("graph-form").addEventListener("submit", (e) => { e.preventDefault(); if ($("graph-id").value.trim()) loadEgo($("graph-id").value); });

/* ═════════════ MODEL PERFORMANCE ═════════════ */
async function loadPerformance() {
  try {
    const d = await api("/api/model-performance");
    const m = d.model_metrics, cv = d.validation.cross_validation, tw = d.validation.threat_type_class_weighting;
    const ifm = d.isolation_forest.isolation_forest;
    renderKpis($("model-kpis"), [
      kpi("blue", "Anomaly (binary)", m.is_anomaly.accuracy * 100, `F1 macro ${fmt(m.is_anomaly.f1_macro, 3)} · CV ${fmt(cv.is_anomaly.mean_f1_macro, 3)} ± ${fmt(cv.is_anomaly.std_f1_macro, 3)}`, 1, "%"),
      kpi("orange", "Risk level (4-class)", m.risk_level.accuracy * 100, `F1 macro ${fmt(m.risk_level.f1_macro, 3)} · CV ${fmt(cv.risk_level.mean_f1_macro, 3)} ± ${fmt(cv.risk_level.std_f1_macro, 3)}`, 1, "%"),
      kpi("red", "Threat type (6-class)", m.threat_type.accuracy * 100, `F1 macro ${fmt(tw.baseline.f1_macro, 3)} &rarr; ${fmt(tw.weighted.f1_macro, 3)} class-weighted`, 1, "%"),
      kpi("purple", "Isolation Forest (unsupervised)", ifm.accuracy * 100, `anomaly-class F1 ${fmt(ifm.f1, 3)} vs XGBoost ${fmt(d.isolation_forest.xgboost_supervised.f1, 3)}`, 1, "%"),
    ]);
    const rc = RISK_ORDER.filter((c) => m.risk_level.classes.includes(c));
    chart("chart-risk-f1", { type: "bar", data: { labels: rc, datasets: [
      { label: "Precision", data: rc.map((c) => m.risk_level.classification_report[c].precision), backgroundColor: "#3d4459", borderRadius: 5 },
      { label: "Recall", data: rc.map((c) => m.risk_level.classification_report[c].recall), backgroundColor: "#818cf8", borderRadius: 5 },
      { label: "F1", data: rc.map((c) => m.risk_level.classification_report[c]["f1-score"]), backgroundColor: rc.map((c) => RISK_COLORS[c]), borderRadius: 5 }] },
      options: { scales: axes({ y: { min: 0, max: 1 }, x: { grid: { display: false } } }) } });
    const tc = tw.per_class.map((r) => r.class);
    chart("chart-threat-f1", { type: "bar", data: { labels: tc.map((c) => nice(c)), datasets: [
      { label: "Baseline", data: tw.per_class.map((r) => r.f1_baseline), backgroundColor: "#3d4459", borderRadius: 5 },
      { label: "Class-weighted", data: tw.per_class.map((r) => r.f1_weighted), backgroundColor: "#6366f1", borderRadius: 5 }] },
      options: { scales: axes({ y: { min: 0, max: 1 }, x: { grid: { display: false }, ticks: { font: { size: 9 } } } }),
        plugins: { tooltip: { callbacks: { afterLabel: (ctx) => `support n=${tw.per_class[ctx.dataIndex].support}` } } } } });
    // confusion matrix (classes in label-encoder order -> reorder to RISK_ORDER)
    const cls = m.risk_level.classes, cm = d.confusion_matrix_risk, idx = rc.map((c) => cls.indexOf(c));
    const mx = Math.max(...cm.flat());
    $("confusion").innerHTML = `<div class="table-wrap"><table class="heat"><thead><tr><th></th>${rc.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>
      ${idx.map((i, a) => `<tr><th>${rc[a]}</th>${idx.map((j) => { const v = cm[i][j];
        return `<td style="background:rgba(99,102,241,${(v / mx * 0.85).toFixed(2)});color:${v / mx > .5 ? "#fff" : "#c7cede"}">${v}</td>`; }).join("")}</tr>`).join("")}
      </tbody></table></div><div class="note">Diagonal = correct. Most errors are between adjacent levels (e.g. MEDIUM vs HIGH).</div>`;
    const ab = d.validation.feature_ablation;
    $("ablation").innerHTML = barRows(ab.map((r) => [r.feature_group, r.f1_drop_from_baseline]), { digits: 4,
      color: (k, v) => v >= 0 ? "linear-gradient(90deg,#f59e0b,#f43f5e)" : "#10d97a", label: (k) => k }) +
      `<div class="note">Baseline macro-F1 ${fmt(d.validation.ablation_baseline_f1, 4)}. Every telemetry source contributes; the synthetic generator
       also ties raw activity volume to risk (the "Activity volume only" row) &mdash; a documented limitation.</div>`;
    $("shap-global").innerHTML = barRows(d.shap_importance.map((r) => [r.feature, r.mean_abs_shap]), { digits: 3, label: (k) => k });
  } catch (e) { fail($("model-kpis"), e); }
}

/* ═════════════ COMPLIANCE ═════════════ */
async function loadCompliance() {
  try {
    const d = await api("/api/compliance");
    renderKpis($("comp-kpis"), [
      kpi("orange", "SoD violators", d.sod_identities, "at least one of 6 rules"),
      kpi("red", "Identities with a control gap", d.identities_with_gap, "NIST SP 800-53 / GDPR Art. 32"),
      kpi("green", "Average compliance score", d.avg_compliance, "100 = no gaps", 1, "/100"),
      kpi("blue", "NIST vs GDPR gaps", d.compliance_by_framework["NIST SP 800-53"],
        `GDPR Art. 32: ${int(d.compliance_by_framework["GDPR Art. 32"])} identities`),
    ]);
    $("sod-by-rule").innerHTML = barRows(Object.entries(d.sod_by_rule), { color: "linear-gradient(90deg,#f59e0b,#fbbf24)" });
    const sev = Object.keys(d.sod_by_severity);
    chart("chart-sod-sev", { type: "doughnut", data: { labels: sev, datasets: [{ data: sev.map((k) => d.sod_by_severity[k]),
      backgroundColor: sev.map((k) => RISK_COLORS[k] || "#3b82f6"), borderColor: "#10141d", borderWidth: 3 }] },
      options: { cutout: "62%", plugins: { legend: { position: "right" } } } });
    $("ctrl-bars").innerHTML = barRows(Object.entries(d.compliance_by_control), { color: "linear-gradient(90deg,#3b82f6,#60a5fa)" });
    const lv = RISK_ORDER.filter((k) => d.score_by_predicted_level[k] !== null);
    chart("chart-comp-level", { type: "bar", data: { labels: lv, datasets: [{ data: lv.map((k) => d.score_by_predicted_level[k]),
      backgroundColor: lv.map((k) => RISK_COLORS[k]), borderRadius: 6 }] },
      options: { plugins: { legend: { display: false } }, scales: axes({ y: { min: 0, max: 100 }, x: { grid: { display: false } } }) } });
    $("worst").innerHTML = d.worst.map((r) => `<tr class="click" data-id="${esc(r.identity_id)}"><td class="mono">${esc(r.identity_id)}</td>
      <td>${esc(r.department)}</td><td>${esc(r.privilege_level)}</td><td class="n">${r.sod_violation_count}</td>
      <td>${esc(nice(r.sod_rules))}</td><td class="n">${fmt(r.compliance_score, 0)}</td><td>${badge(r.xgb_risk_level_pred)}</td></tr>`).join("");
  } catch (e) { fail($("comp-kpis"), e); }
}

/* ═════════════ ORGANIZATION ═════════════ */
async function loadOrg() {
  try {
    const d = await api("/api/org");
    const deps = d.departments;
    chart("chart-dept", { type: "bar", data: { labels: deps.map((x) => x.department), datasets: [
      { label: "Avg model risk (0-100)", data: deps.map((x) => x.avg_model_risk), backgroundColor: "#f59e0b", borderRadius: 5, yAxisID: "y" },
      { label: "True anomaly rate %", data: deps.map((x) => x.true_anomaly_rate * 100), backgroundColor: "#f43f5e", borderRadius: 5, yAxisID: "y" },
      { label: "Admin %", data: deps.map((x) => x.pct_admin * 100), backgroundColor: "#6366f1", borderRadius: 5, yAxisID: "y" }] },
      options: { scales: axes({ x: { grid: { display: false } } }) } });
    const mt = d.metrics;
    $("chi2-note").textContent = `chi-square department vs anomaly: p = ${mt.chi2_p_value} (${mt.chi2_p_value >= 0.05 ? "no significant dependence" : "significant"})`;
    $("dept-table").innerHTML = deps.map((x) => `<tr><td>${esc(x.department)}</td><td class="n">${int(x.identities)}</td>
      <td class="n">${fmt(x.avg_model_risk)}</td><td class="n">${pct(x.pct_admin)}</td><td class="n">${pct(x.pct_mfa)}</td>
      <td class="n">${fmt(x.avg_compliance)}</td><td>${esc(x.flagged_metrics || "")}</td></tr>`).join("");
    $("peer-note").textContent = `${mt.peer_outliers} outliers · ${pct(mt.peer_outlier_true_anomaly_rate, 0)} true anomalies (base ${pct(mt.base_anomaly_rate, 0)})`;
    $("peer-table").innerHTML = d.peer_outliers.map((x) => `<tr class="click" data-id="${esc(x.identity_id)}"><td class="mono">${esc(x.identity_id)}</td>
      <td>${esc(x.department)}</td><td class="n">${fmt(x.max_peer_z)}</td><td style="font-size:11.5px">${esc(x.peer_outlier_features)}</td>
      <td class="n">${fmt(x.xgb_risk_index)}</td></tr>`).join("");
  } catch (e) { fail($("chi2-note"), e); }
}

/* ═════════════ QUIET RISK ═════════════ */
async function loadQuietRisk() {
  try {
    const d = await api("/api/quiet-risk");
    const m = d.metrics, byQ = Object.fromEntries(m.quadrants.map((q) => [q.quadrant, q]));
    const meta = { QUIET_RISK: ["purple", "behaviour low · structure high"], CONSENSUS_RISK: ["red", "both layers high"],
      NOISY_BENIGN: ["orange", "behaviour high · structure low"], CONSENSUS_BENIGN: ["green", "both layers low"] };
    renderKpis($("quadrant-kpis"), Object.keys(meta).map((q) => kpi(meta[q][0], nice(q), m.quadrant_counts[q] || 0,
      `${meta[q][1]}<br>true anomalies ${pct(byQ[q]?.true_anomaly_rate)} · blast ${fmt(byQ[q]?.mean_blast_radius)} · compliance ${fmt(byQ[q]?.mean_compliance)}`)));
    const groups = Object.keys(QUAD_COLORS);
    chart("chart-quadrants", { type: "scatter", data: { datasets: groups.map((q) => ({ label: nice(q),
      data: d.scatter.filter((p) => p.quadrant === q).map((p) => ({ x: p.behavioural_pct, y: p.structural_pct, id: p.identity_id })),
      backgroundColor: QUAD_COLORS[q] + "b0", pointRadius: q === "QUIET_RISK" ? 3.5 : 2.2 })) },
      options: { scales: axes({ x: { min: 0, max: 100, title: { display: true, text: "Behavioural percentile (model risk)" } },
        y: { min: 0, max: 100, title: { display: true, text: "Structural percentile (blast + SoD + compliance)" } } }),
        plugins: { tooltip: { callbacks: { label: (c) => `${c.raw.id}: behaviour ${fmt(c.raw.x, 0)}, structure ${fmt(c.raw.y, 0)}` } } },
        onClick: (_, els) => { if (els.length) { const ds = charts["chart-quadrants"].data.datasets[els[0].datasetIndex]; openIdentity(ds.data[els[0].index].id); } } } });
    $("quiet-rules").innerHTML = barRows(Object.entries(m.quiet_top_sod_rules), { color: "linear-gradient(90deg,#a855f7,#c084fc)" });
    $("quiet-note").innerHTML = `<b>${int(m.quiet_risk_count)}</b> identities (${fmt(m.quiet_risk_pct)}%) are rated LOW/MEDIUM by the behavioural model while
      being structurally in the top 20%. ${pct(m.quiet_reach_critical_rate, 0)} of them can reach a CRITICAL resource. Their ground-truth anomaly
      rate is ${pct(m.quiet_true_anomaly_rate)} vs ${pct(m.consensus_benign_true_anomaly_rate)} for consensus-benign identities: the risk is latent
      (what they <i>could</i> do), which is exactly why a structural layer is needed.`;
    $("quiet-table").innerHTML = d.top.map((r) => `<tr class="click" data-id="${esc(r.identity_id)}"><td class="mono">${esc(r.identity_id)}</td>
      <td>${esc(r.department)}</td><td>${esc(r.privilege_level)}</td><td>${r.mfa_enabled ? "yes" : "<b style='color:#fda4af'>NO</b>"}</td>
      <td>${badge(r.xgb_risk_level_pred)}</td><td class="n">${fmt(r.blast_radius_score)}</td><td class="n">${r.sod_severity_score}</td>
      <td style="font-size:11.5px">${esc(nice(r.sod_rules))}</td><td class="n">${fmt(r.compliance_score, 0)}</td><td class="n">${fmt(r.disagreement_score)}</td></tr>`).join("");
  } catch (e) { fail($("quadrant-kpis"), e); }
}

/* ═════════════ REMEDIATION ═════════════ */
async function loadRemediation() {
  try {
    const d = await api("/api/remediation");
    const m = d.metrics, S = m.strategies;
    const best = (k) => Math.max(...S.map((s) => s[k]));
    $("strategy-cards").innerHTML = S.map((s, i) => `<div class="card" style="border-color:${STRAT_COLORS[i]}55">
      <div class="card-title"><span style="color:${STRAT_COLORS[i] === "#5b6478" ? "#c7cede" : STRAT_COLORS[i]}">${esc(s.strategy)}</span><small>${s.identities_fixed} fixes</small></div>
      <div class="grid grid-3" style="gap:10px">
        ${[["Risk reduction", s.total_risk_reduction, "total_risk_reduction"], ["Attack risk removed", s.total_ear_removed, "total_ear_removed"],
          ["Critical paths", s.critical_paths_removed, "critical_paths_removed"]].map(([t, v, k]) =>
          `<div><div class="note" style="margin:0">${t}</div><div style="font-size:20px;font-weight:800;font-family:var(--mono);color:${v === best(k) ? "#6ee7b7" : "#f4f6fa"}">${int(v)}</div></div>`).join("")}
      </div>
      <div class="note">${fmt(s.pct_org_ear_removed)}% of the organisation's expected attack risk · ${s.quiet_risk_identities_fixed} quiet-risk identities fixed</div>
      <div class="chips">${Object.entries(s.action_mix).map(([a, n]) => `<span class="chip">${esc(nice(a))}: <b>${n}</b></span>`).join("")}</div></div>`).join("");
    const labels = S.map((s) => s.strategy.split(" (")[0]);
    [["chart-rem-risk", "total_risk_reduction"], ["chart-rem-ear", "total_ear_removed"], ["chart-rem-crit", "critical_paths_removed"]].forEach(([id, k]) =>
      chart(id, { type: "bar", data: { labels, datasets: [{ data: S.map((s) => s[k]), backgroundColor: STRAT_COLORS, borderRadius: 7 }] },
        options: { plugins: { legend: { display: false } }, scales: axes({ x: { grid: { display: false } } }) } }));
    $("strategy-tables").innerHTML = d.top_by_strategy.map(({ strategy: s, rows }, i) => `<div class="card">
      <div class="card-title" style="color:${STRAT_COLORS[i] === "#5b6478" ? "#c7cede" : STRAT_COLORS[i]}">${esc(s)} <small>first 10 of ${m.budget}</small></div>
      <div class="table-wrap"><table><thead><tr><th>Identity</th><th>Action</th><th class="n">&Delta; risk</th><th class="n">&Delta; attack</th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="click" data-id="${esc(r.identity_id)}"><td class="mono">${esc(r.identity_id)}</td><td style="font-size:11.5px">${esc(nice(r.action))}</td>
        <td class="n">${fmt(r.risk_reduction)}</td><td class="n">${fmt(r.ear_reduction, 2)}</td></tr>`).join("")}</tbody></table></div></div>`).join("");
    $("rem-summary").textContent = d.summary_text;
  } catch (e) { fail($("strategy-cards"), e); }
}

/* ═════════════ RESEARCH ═════════════ */
async function loadResearch() {
  try {
    const d = await api("/api/research");
    const f = d.fusion, fi = d.fidelity, g = d.graph, ag = fi.explanation_action_agreement;
    const gain = f.f1_gain_vs_best_single, ci = f.f1_gain_ci95;
    renderKpis($("research-hero"), [
      kpi(f.gain_significant ? "green" : "orange", "Gap 1 · fusion F1 gain", gain * 100,
        `points vs ${esc(f.best_single_signal)} · 95% CI [${fmt(ci[0] * 100, 2)}, ${fmt(ci[1] * 100, 2)}] · ${f.gain_significant ? "significant" : "not significant"}`, 2, " pts"),
      kpi("green", "Gap 2 · SHAP fidelity", fi.global_drop_ratio,
        `x the drop of random features · ${fmt(fi.pct_high_fidelity)}% high-fidelity (${pct(fi.pct_high_fidelity_by_predicted_class["1"], 0)} of predicted anomalies)`, 1, "x"),
      kpi("blue", "Explanation-action agreement", ag ? ag.agreement_rate * 100 : 0,
        ag ? `of ${int(ag.identities)} recommended actions touch the identity's own top SHAP factors` : "not computed", 1, "%"),
      kpi("purple", "Gap 3 · novel graph outliers", g.novel_outliers,
        `${int(g.structural_outliers)} structural outliers · neighbour signal ${g.neighbour_signal_adds_value ? "adds" : "adds no"} AUC (${g.test_auc_own_risk} &rarr; ${g.test_auc_own_plus_neighbour})`),
    ]);
    $("fusion-table").innerHTML = f.results.map((r) => `<tr style="${r.signal.startsWith("FUSION") ? "font-weight:700" : ""}">
      <td>${esc(r.signal)}</td><td class="n">${fmt(r.precision, 3)}</td><td class="n">${fmt(r.recall, 3)}</td>
      <td class="n">${fmt(r.f1, 3)}</td><td class="n">${fmt(r.roc_auc, 3)}</td></tr>`).join("");
    $("fusion-note").innerHTML = `Meta-model weights: ${Object.entries(f.meta_coefficients).map(([k, v]) => `${esc(k)} <b>${fmt(v, 2)}</b>`).join(" · ")}.
      The temporal signal carries no information on this synthetic data (events are spread uniformly over the year), so fusion cannot beat XGBoost here.`;
    chart("chart-fidelity", { type: "line", data: { labels: fi.curve.map((c) => "k=" + c.k), datasets: [
      { label: "Top-k SHAP features removed", data: fi.curve.map((c) => c.mean_drop_shap_topk), borderColor: "#818cf8", backgroundColor: "rgba(99,102,241,.15)", fill: true, tension: .3 },
      { label: "k random features removed", data: fi.curve.map((c) => c.mean_drop_random_k), borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,.1)", fill: true, tension: .3 }] },
      options: { scales: axes({ y: { title: { display: true, text: "Mean probability drop" } } }) } });
    $("research-texts").innerHTML = Object.entries(d.texts).filter(([, t]) => t).map(([k, t]) =>
      `<div class="card"><div class="card-title">${esc(k)} <small>full report file</small></div><pre class="summary-text">${esc(t)}</pre></div>`).join("");
  } catch (e) { fail($("research-hero"), e); }
}

const LOADERS = { overview: loadOverview, explorer: () => loadIdentities(1), graph: loadGraph, performance: loadPerformance,
  compliance: loadCompliance, org: loadOrg, quietrisk: loadQuietRisk, remediation: loadRemediation, research: loadResearch };

/* ── init ── */
loadDepartments().catch(() => {});
const pageFromHash = () => { const h = (location.hash || "#overview").slice(1); return LOADERS[h] ? h : "overview"; };
showPage(pageFromHash());
window.addEventListener("hashchange", () => showPage(pageFromHash()));
