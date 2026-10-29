/* AgroVision AI — frontend logic. Vanilla JS, no dependencies, works offline.
   Talks to the FastAPI backend with relative URLs: /api/health, /api/crops, /api/samples, /api/analyze, /api/metrics.
   Every server field may be missing or null, so all rendering goes through the small helpers below,
   and every server string inserted as HTML passes through esc(). */
'use strict';

const MAX_BYTES = 10 * 1024 * 1024;
const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const STEPS = ['Quality gate', 'Plant / insect check', 'Crop & diagnosis', 'Nutrient analysis', 'Pest detection', 'Advisory'];
const state = { file: null, url: null, busy: false, health: null, crops: null, samples: [], sampleKind: 'all', tables: {} };

// ---------- helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const arr = v => (Array.isArray(v) ? v : []);
const obj = v => (v && typeof v === 'object' && !Array.isArray(v) ? v : {});
const fix = (v, d = 1) => (isNum(v) ? v.toFixed(d) : '—');
const pct = (v, d = 1) => (isNum(v) ? (v * 100).toFixed(d) + '%' : '—'); // 0..1 -> "97.0%"
const pctRaw = (v, d = 1) => (isNum(v) ? v.toFixed(d) + '%' : '—'); // already 0..100
const num = v => (isNum(v) ? v.toLocaleString('en-IN') : '—');
const clamp01 = v => (isNum(v) ? Math.min(1, Math.max(0, v)) : 0);
const txt = (v, fallback = '—') => esc(v == null || v === '' ? fallback : v);
const blank = v => v == null || /^[-—–\s]*$/.test(String(v)); // placeholders like "-" or "—"
const safeUrl = u => (typeof u === 'string' && !/^\s*(javascript:|vbscript:|data:(?!image\/))/i.test(u) ? u : '');
const fmtBytes = b => (b < 1024 * 1024 ? (b / 1024).toFixed(0) + ' KB' : (b / 1048576).toFixed(1) + ' MB');
const scrollToEl = el => el && el.scrollIntoView({ behavior: REDUCED_MOTION ? 'auto' : 'smooth', block: 'start' });
const splitLabel = l => { // "Tomato · Early Blight" -> ["Tomato", "Early Blight"]
  const s = typeof l === 'string' ? l : '';
  const i = s.indexOf(' · ');
  return i < 0 ? [s, ''] : [s.slice(0, i), s.slice(i + 3)];
};
const ci = c => (Array.isArray(c) && isNum(c[0]) && isNum(c[1]) ? `95% CI ${pct(c[0], 1)} – ${pct(c[1], 1)}` : '');

// Kind of finding: label, badge tone and chart colour. "other" = the router classes (not a plant / insect photo).
const KINDS = {
  disease: ['Disease', 'bad', '#d9443a'], pest: ['Pest', 'violet', '#8a4fc2'], deficiency: ['Deficiency', 'warn', '#e39a2d'],
  healthy: ['Healthy', 'ok', '#2f9e44'], abiotic: ['Abiotic', 'info', '#3b82c4'], insect: ['Insect', 'teal', '#14968b'],
  other: ['Router', 'neutral', '#a3aea8'],
};
const KIND_ORDER = ['disease', 'pest', 'deficiency', 'healthy', 'abiotic'];
const kindInfo = k => KINDS[k] || [k ? String(k) : 'Other', 'neutral', '#a3aea8'];
const kindBadge = k => badge(kindInfo(k)[0], kindInfo(k)[1]);

// Colour tone for common categorical values coming from the API.
const TONE = {
  Low: 'ok', None: 'ok', Routine: 'ok', Healthy: 'ok', Normal: 'ok',
  Moderate: 'warn', Medium: 'warn', Soon: 'warn', 'Mild stress': 'warn', 'Moderate stress': 'warn',
  High: 'bad', Immediate: 'bad', Severe: 'bad',
};
const TYPE_TONE = { Fungal: 'amber', Oomycete: 'teal', Bacterial: 'violet', Viral: 'bad', Mite: 'warn', Insect: 'warn', Algal: 'ok', Phytoplasma: 'info' };
const TREAT_TONE = { Fungicide: 'amber', Bactericide: 'violet', Insecticide: 'bad', Miticide: 'warn', Nematicide: 'violet', Cultural: 'info', Biological: 'ok', Fertilizer: 'teal' };
const PEST_SOURCE = { classifier: ['Learned (classifier)', 'info'], 'disease-vector': ['Disease vector', 'violet'], 'damage-pattern': ['Damage pattern (rules)', 'amber'] };
const NUTRIENT_SOURCE = { model: ['Learned from labelled images', 'info'], rules: ['Colour-rule estimate', 'amber'] };
const PROFILE = [
  ['healthy_green_pct', 'Healthy green', '#2f9e44'], ['chlorosis_pct', 'Chlorosis (yellow)', '#e9c33b'],
  ['necrosis_pct', 'Necrosis (brown)', '#9c5a2c'], ['dark_lesion_pct', 'Dark lesions', '#3a2a22'],
  ['purpling_pct', 'Purpling', '#7e3fa3'], ['pale_pct', 'Pale / whitish', '#d8d3c4'], ['other_pct', 'Other', '#a3aea8'],
];
// Colours drawn by backend/vision.py _overlay() — keep in sync. Fills are tints; outlines are rings.
const SEG_LEGEND = `<ul class="legend legend-inline">${[
  ['#ff2828', 'Necrosis'], ['#820000', 'Dark lesions'], ['#ffe100', 'Chlorosis'], ['#aa00ff', 'Purpling'],
  ['#00ffff', 'Stippling specks'], ['#ff00ff', 'Holes (outline)', 1], ['#00ff78', 'Leaf outline', 1],
].map(([c, l, ring]) => `<li><i style="${ring ? 'border:2px solid ' + c : 'background:' + c}"></i>${l}</li>`).join('')}</ul>`;
const HEALTH_FALLBACK = { Healthy: '#2f9e44', 'Mild stress': '#e0a82e', 'Moderate stress': '#e07b2e', Severe: '#c53030' };

const confTone = v => (!isNum(v) ? '' : v >= 0.8 ? 'b-ok' : v >= 0.5 ? 'b-warn' : 'b-bad');
const confLevel = v => (!isNum(v) ? '' : v >= 0.8 ? 'High' : v >= 0.5 ? 'Moderate' : 'Low');
const levelBar = l => ({ High: 'b-bad', Moderate: 'b-warn' })[l] || 'b-ok';
const riskText = r => (r === 'None' ? 'No risk' : `${r} risk`); // API risk: None | Low | Medium | High
const badge = (text, tone) => `<span class="badge badge-${tone || TONE[text] || 'neutral'}">${esc(text)}</span>`;
const bar = (v, cls = '') => `<div class="bar ${cls}"><span style="width:${(clamp01(v) * 100).toFixed(1)}%"></span></div>`;
// labelHtml must already be escaped; meter() is the plain-text version.
const meterHtml = (labelHtml, v, cls = '', value = pct(v), note = '') =>
  `<div class="meter"><div class="meter-head"><span>${labelHtml}</span><span class="num">${esc(value)}</span></div>${bar(v, cls)}${
    note ? `<p class="meter-note">${esc(note)}</p>` : ''}</div>`;
const meter = (label, ...rest) => meterHtml(esc(label), ...rest);
const bullets = items => (arr(items).length ? `<ul class="list">${arr(items).map(i => `<li>${esc(i)}</li>`).join('')}</ul>` : '');
const tags = items => (arr(items).length ? `<ul class="tags">${arr(items).map(i => `<li>${esc(i)}</li>`).join('')}</ul>` : '');
// Result card. num/title are static strings from this file (not server data).
const card = (num, title, body, cls = '') =>
  `<article class="card rcard ${cls}"><h3>${num ? `<span class="obj-num">${num} ·</span>` : ''}${title}</h3>${body}</article>`;
// Proportional bar + legend. parts: [{label, color, v}] (any unit), fmt formats one value for the legend.
function stackBar(parts, fmt) {
  parts = parts.filter(p => isNum(p.v) && p.v > 0);
  const total = parts.reduce((s, p) => s + p.v, 0);
  if (!total) return '';
  return `<div class="stack" role="img" aria-label="${esc(parts.map(p => `${p.label} ${fmt(p.v)}`).join(', '))}">
      ${parts.map(p => `<span style="width:${((p.v / total) * 100).toFixed(2)}%;background:${p.color}"></span>`).join('')}</div>
    <ul class="legend">${parts.map(p => `<li><i style="background:${p.color}"></i>${esc(p.label)}<b>${esc(fmt(p.v))}</b></li>`).join('')}</ul>`;
}

// ---------- network ----------
async function api(url, opts) {
  let res;
  try { res = await fetch(url, opts); } catch { throw new Error('Cannot reach the server. Is the backend running?'); }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) {
    const d = body && body.detail;
    const msg = typeof d === 'string' ? d : Array.isArray(d) ? d.map(x => x && x.msg).filter(Boolean).join('; ') : '';
    throw new Error(`Server error ${res.status}${msg ? ': ' + msg : ''}`);
  }
  if (body == null) throw new Error('The server returned an unreadable response.');
  return body;
}

let toastTimer;
function toast(msg, kind = 'info') {
  const t = $('#toast');
  t.textContent = msg;
  t.className = `toast no-print toast-${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 7000);
}

async function checkHealth() {
  const pill = $('#apiStatus');
  let cls = 'pill-bad', label = 'Offline', title = 'Backend not reachable';
  try {
    const h = obj(await api('/api/health'));
    state.health = h;
    const ok = h.status === 'ok' && h.model_loaded !== false;
    cls = ok ? 'pill-ok' : 'pill-warn';
    label = ok ? 'Model online' : 'Model not loaded';
    state.device = h.device ? String(h.device).toUpperCase() : null;
    title = [h.model, state.device, h.classes && `${h.classes} classes`, h.pest_model && `insect model ${h.pest_model}`]
      .filter(Boolean).join(' · ') || label;
    renderHeroStats();
  } catch { /* stays offline */ }
  if ($('.pill-text', pill).textContent !== label) $('.pill-text', pill).textContent = label; // avoid re-announcing
  pill.className = `pill ${cls}`;
  pill.title = title;
}

// "38 crops · 221 conditions · 102 pest species", all counted from the API.
function renderHeroStats() {
  const h = obj(state.health), crops = arr(state.crops);
  const nCrops = crops.length || h.crops;
  const nConditions = crops.reduce((s, c) => s + arr(obj(c).conditions).length, 0);
  const parts = [[nCrops, 'crops'], [nConditions, 'conditions'], [h.pest_species, 'pest species']].filter(([n]) => isNum(n) && n > 0);
  const el = $('#heroStats');
  el.innerHTML = parts.map(([n, label]) => `<span><b>${esc(num(n))}</b> ${label}</span>`).join('');
  el.hidden = !parts.length;
}

// ---------- samples ----------
const SAMPLE_FILTERS = [['all', 'All'], ['disease', 'Disease'], ['pest', 'Pest'], ['deficiency', 'Deficiency'], ['healthy', 'Healthy'], ['insect', 'Insect']];

async function loadSamples() {
  const box = $('#samples');
  try {
    state.samples = arr(await api('/api/samples')).map(obj).filter(s => safeUrl(s.url));
    renderSampleFilters();
    renderSamples();
  } catch {
    box.innerHTML = '<p class="muted small">Samples are unavailable while the backend is offline.</p>';
  }
  box.removeAttribute('aria-busy');
}

function renderSampleFilters() {
  const count = k => state.samples.filter(s => k === 'all' || s.kind === k).length;
  $('#sampleFilters').innerHTML = SAMPLE_FILTERS.map(([k, label]) => {
    const n = count(k);
    return `<button type="button" class="fchip${k === 'all' ? '' : ' kind-' + k}" data-sample-kind="${k}" aria-pressed="${state.sampleKind === k}"${n ? '' : ' disabled'}>
      ${label}<span class="fchip-n">${n}</span></button>`;
  }).join('');
}

function renderSamples() {
  const list = state.samples.filter(s => state.sampleKind === 'all' || s.kind === state.sampleKind);
  $('#samples').innerHTML = list.length ? list.map(s => {
    const [crop, condition] = splitLabel(s.label);
    return `<button type="button" class="sample" data-sample="${esc(s.url)}" data-label="${esc(s.label)}" title="${esc(s.label)}${s.imagery ? ` (${esc(s.imagery)} photo)` : ''}">
      <span class="sample-img"><img src="${esc(s.url)}" alt="" loading="lazy" width="120" height="120">${s.imagery ? `<span class="imagery">${esc(s.imagery)}</span>` : ''}</span>
      <span class="sample-cap"><b>${txt(crop, 'Sample')}</b>${condition ? `<small>${esc(condition)}</small>` : ''}${s.kind ? kindBadge(s.kind) : ''}</span></button>`;
  }).join('') : '<p class="muted small">No samples of this kind yet.</p>';
}

// ---------- supported crops ----------
async function loadCrops() {
  const grid = $('#cropGrid');
  try {
    state.crops = arr(await api('/api/crops')).map(obj).filter(c => c.crop);
  } catch (e) {
    grid.innerHTML = `<div class="card empty"><span>The crop list is unavailable right now (${esc(e.message)}).</span>
      <button type="button" class="btn btn-ghost" data-action="crops">Retry</button></div>`;
    return;
  }
  const counts = {};
  state.crops.forEach(c => arr(c.conditions).forEach(x => (counts[obj(x).kind] = (counts[obj(x).kind] || 0) + 1)));
  $('#cropLegend').innerHTML = KIND_ORDER.filter(k => counts[k]).map(k =>
    `<span class="kchip kind-${k}">${kindInfo(k)[0]}<b>${counts[k]}</b></span>`).join('');
  renderHeroStats();
  renderCrops();
}

function renderCrops() {
  if (!state.crops) return;
  const q = $('#cropSearch').value.trim().toLowerCase();
  const hit = s => String(s ?? '').toLowerCase().includes(q);
  const list = q ? state.crops.filter(c => [c.crop, c.scientific_name, c.family, ...arr(c.conditions).map(x => obj(x).name)].some(hit)) : state.crops;
  $('#cropCount').textContent = q ? `${list.length} of ${state.crops.length} crops match` : `${state.crops.length} crops`;
  $('#cropGrid').innerHTML = list.length ? list.map(c => {
    const conds = arr(c.conditions).map(obj).sort((a, b) =>
      KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) || String(a.name).localeCompare(String(b.name)));
    const counts = KIND_ORDER.map(k => [k, conds.filter(x => x.kind === k).length]).filter(([, n]) => n);
    return `<article class="card crop-card">
      <h3>${txt(c.crop)}</h3>
      <p class="crop-sci">${c.scientific_name ? `<i>${esc(c.scientific_name)}</i>` : ''}${c.scientific_name && c.family ? ' · ' : ''}${c.family ? esc(c.family) : ''}</p>
      <p class="crop-counts"><b>${conds.length}</b> condition${conds.length === 1 ? '' : 's'}${counts.map(([k, n]) =>
        ` <span class="kcount"><i class="kdot kind-${k}" aria-hidden="true"></i>${n} ${kindInfo(k)[0].toLowerCase()}</span>`).join('')}</p>
      <ul class="cond-chips">${conds.map(x => `<li class="kind-${esc(x.kind)}${q && hit(x.name) ? ' hit' : ''}" title="${esc(kindInfo(x.kind)[0])}">${txt(x.name)}<span class="visually-hidden"> (${esc(kindInfo(x.kind)[0])})</span></li>`).join('')}</ul>
    </article>`;
  }).join('') : `<div class="card empty">No crop or condition matches &ldquo;${esc(q)}&rdquo;.</div>`;
}

// ---------- file selection ----------
function pickFile(file) {
  if (!file) return false;
  const isImage = (file.type || '').startsWith('image/') || /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(file.name || '');
  if (!isImage) { toast('Please choose an image file (JPG, PNG or WEBP).', 'error'); return false; }
  if (!file.size) { toast('That file is empty.', 'error'); return false; }
  if (file.size > MAX_BYTES) { toast(`That image is ${fmtBytes(file.size)} — the limit is 10 MB.`, 'error'); return false; }
  hideOutputs();
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file;
  state.url = URL.createObjectURL(file);
  $('#previewImg').src = state.url;
  $('#fileName').textContent = file.name || 'photo';
  $('#fileSize').textContent = fmtBytes(file.size);
  $('#dzEmpty').hidden = true;
  $('#dzPreview').hidden = false;
  $('#analyzeBtn').disabled = false;
  $('#changeBtn').hidden = false;
  $('#clearBtn').hidden = false;
  return true;
}

async function useSample(url, label) {
  if (state.busy) return;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error();
    const blob = await res.blob();
    const name = decodeURIComponent(url.split('/').pop() || 'sample.jpg');
    if (pickFile(new File([blob], name, { type: blob.type || 'image/jpeg' }))) analyze();
  } catch {
    toast(`Couldn't load the sample "${label || url}".`, 'error');
  }
}

function hideOutputs() {
  $('#results').hidden = true;
  $('#rejected').hidden = true;
  $('#progress').hidden = true;
}

function resetAll() {
  if (state.busy) return;
  hideOutputs();
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = null;
  state.url = null;
  $('#fileInput').value = '';
  $('#previewImg').removeAttribute('src');
  $('#dzEmpty').hidden = false;
  $('#dzPreview').hidden = true;
  $('#analyzeBtn').disabled = true;
  $('#changeBtn').hidden = true;
  $('#clearBtn').hidden = true;
  scrollToEl($('#analyze'));
}

// ---------- analysis ----------
function setBusy(busy) {
  state.busy = busy;
  const btn = $('#analyzeBtn');
  btn.disabled = busy || !state.file;
  $('.btn-label', btn).textContent = busy ? 'Analyzing…' : 'Analyze photo';
  $('#samples').inert = busy;
  $('#sampleFilters').inert = busy;
  $('#changeBtn').disabled = busy;
}

// Animates the pipeline stages in parallel with the request (~1 s in total); returns finish(outcome).
// Results appear as soon as both the response and this short animation are done.
function runStepper() {
  const items = $$('#stepper li');
  const text = $('#progressText');
  const stepMs = REDUCED_MOTION ? 0 : 150;
  const minShown = sleep(stepMs * (items.length + 1)); // one timer, so a throttled tab can't stretch it per step
  let i = 0, responded = false;
  const mark = () => {
    items.forEach((li, k) => (li.dataset.state = k < i ? 'done' : k === i ? 'active' : ''));
    text.textContent = items[i] ? `Step ${i + 1} of ${items.length}: ${STEPS[i]}…` : 'Analysis complete.';
  };
  $('#progress').hidden = false;
  mark();
  // the last stage stays active until the response arrives, then ticks to "all done"
  const timer = setInterval(() => { if (i < items.length - 1 || (responded && i < items.length)) { i++; mark(); } }, stepMs || 1);
  return async outcome => {
    if (outcome === 'ok') {
      responded = true;
      await minShown;
      clearInterval(timer);
      i = items.length;
      mark();
      return;
    }
    clearInterval(timer);
    if (outcome === 'error') return; // network/server error: caller hides the stepper and shows a toast
    const failed = outcome === 'not-plant' ? 1 : 0; // which stage rejected the photo
    items.forEach((li, k) => (li.dataset.state = k < failed ? 'done' : k === failed ? 'fail' : ''));
    text.textContent = failed ? 'Photo rejected: it does not look like a plant leaf.' : 'Photo rejected by the quality gate.';
    await sleep(REDUCED_MOTION ? 0 : 350);
  };
}

const failedPlantCheck = q => arr(obj(q).checks).some(c => obj(c).name === 'Plant check' && obj(c).passed === false);

async function analyze(force = false) {
  if (!state.file || state.busy) return;
  setBusy(true);
  $('#results').hidden = true;
  $('#rejected').hidden = true;
  const finish = runStepper();
  $('#progress').scrollIntoView({ behavior: REDUCED_MOTION ? 'auto' : 'smooth', block: 'center' });
  try {
    const fd = new FormData();
    fd.append('file', state.file, state.file.name || 'photo.jpg');
    const data = obj(await api('/api/analyze' + (force ? '?force=true' : ''), { method: 'POST', body: fd }));
    const rejected = data.status === 'rejected';
    await finish(!rejected ? 'ok' : failedPlantCheck(data.quality) ? 'not-plant' : 'rejected');
    $('#progress').hidden = true;
    if (rejected) renderRejected(data);
    else if (data.mode === 'insect') renderInsect(data);
    else renderResults(data);
  } catch (e) {
    await finish('error');
    $('#progress').hidden = true;
    toast(e.message || 'Something went wrong while analyzing the image.', 'error');
  } finally {
    setBusy(false);
  }
}

// ---------- rejected (quality gate / plant check) ----------
function checksTable(checks) {
  if (!checks.length) return '';
  const rows = checks.map(c => {
    c = obj(c);
    const tone = c.passed ? 'ok' : c.severity === 'hard' ? 'bad' : 'warn';
    const label = c.passed ? 'Pass' : c.severity === 'hard' ? 'Fail' : 'Warning';
    const value = c.name === 'Plant check' ? (c.passed ? 'Plant' : 'Not a plant') // the value of this check is a yes/no
      : isNum(c.value) ? fix(c.value, Number.isInteger(c.value) ? 0 : Math.abs(c.value) < 10 ? 2 : 1) : txt(c.value);
    return `<tr class="${c.passed ? '' : 'row-' + tone}">
      <td><strong>${txt(c.name)}</strong>${c.message && c.message !== 'OK' ? `<div class="small muted">${esc(c.message)}</div>` : ''}</td>
      <td class="num">${value}</td><td class="small">${txt(c.threshold)}</td><td>${badge(label, tone)}</td></tr>`;
  }).join('');
  return `<div class="table-wrap"><table class="table"><thead><tr><th scope="col">Check</th><th scope="col" class="num">Value</th>
    <th scope="col">Threshold</th><th scope="col">Result</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderRejected(d) {
  const q = obj(d.quality);
  const rank = c => (obj(c).passed ? 2 : obj(c).severity === 'hard' ? 0 : 1); // hard failures first
  const checks = arr(q.checks).slice().sort((a, b) => rank(a) - rank(b));
  const notPlant = failedPlantCheck(q);
  const el = $('#rejected');
  el.innerHTML = `
    <div class="rej-head">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 1 21h22L12 2zm0 6 .01 0L12 15m0 3v.01" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <div class="rej-text">
        <h2>${notPlant ? 'This doesn&rsquo;t look like a plant leaf' : 'This photo didn&rsquo;t pass the photo checks'}</h2>
        <p class="muted">${notPlant
          ? 'The leaf model checks every photo before diagnosing it. This one looks like neither a crop leaf nor an insect pest, so no diagnosis was made.'
          : `Photo check score <strong>${isNum(q.score) ? q.score : '—'}/100</strong>. See the failed checks below.`}
          A clear photo of a single leaf &mdash; or of the insect itself &mdash; filling most of the frame gives a much more reliable result.</p>
      </div>
      ${state.url ? `<img class="rej-thumb" src="${esc(state.url)}" alt="The rejected photo">` : ''}
    </div>
    ${arr(q.issues).length ? `<div class="callout callout-warn"><strong>Problems found</strong>${bullets(q.issues)}</div>` : ''}
    ${checksTable(checks)}
    ${arr(q.tips).filter(Boolean).length ? `<div class="tips"><h3>How to retake the photo</h3>${bullets(arr(q.tips).filter(Boolean))}</div>` : ''}
    <div class="actions">
      <button type="button" class="btn btn-primary" data-action="choose">Choose another photo</button>
      <button type="button" class="btn btn-warn" data-action="force">Analyze anyway</button>
    </div>
    <p class="small muted" style="margin-top:10px">&ldquo;Analyze anyway&rdquo; ${notPlant
      ? 'forces a leaf diagnosis &mdash; for anything other than a plant the result is not meaningful.'
      : 'skips the quality gate &mdash; results on a poor photo may be unreliable.'}</p>`;
  el.hidden = false;
  el.focus({ preventScroll: true });
  scrollToEl(el);
}

// ---------- results: shared pieces ----------
function showResults(html, gaugeValue) {
  const el = $('#results');
  el.innerHTML = html;
  el.hidden = false;
  animateGauge(gaugeValue);
  el.focus({ preventScroll: true });
  scrollToEl(el);
}

function printHead(d, title) {
  return `<div class="print-only print-head"><h1>AgroVision AI — ${title}</h1>
    <p class="small">${esc(new Date().toLocaleString())} · Photo: ${txt(state.file && state.file.name)} · Request ${txt(d.request_id)}</p></div>`;
}

const resultFooter = (d, again) => `${d.disclaimer ? `<p class="disclaimer"><strong>Disclaimer:</strong> ${esc(d.disclaimer)}</p>` : ''}
  <div class="actions results-actions">
    <button type="button" class="btn btn-primary" data-action="reset">${again}</button>
    <button type="button" class="btn btn-ghost" data-action="print">Print report</button>
  </div>`;

const warningsBanner = d => (arr(d.warnings).length ? `<div class="banner" role="note"><strong>Please note</strong>${bullets(d.warnings)}</div>` : '');

// Ring gauge (value 0..100); animateGauge() fills it in.
function gaugeBox(value, color, unit, aria, category, caption) {
  const C = (2 * Math.PI * 52).toFixed(2);
  return `<div class="gauge-box">
    <div class="gauge" style="--g:${color}" role="img" aria-label="${esc(aria)}">
      <svg viewBox="0 0 120 120" aria-hidden="true"><circle class="g-track" cx="60" cy="60" r="52"/>
        <circle class="g-arc" id="gArc" cx="60" cy="60" r="52" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg>
      <div class="g-center" aria-hidden="true"><span class="g-num" id="gNum">${isNum(value) ? 0 : '—'}</span><span class="g-max">${unit}</span></div>
    </div>
    <span class="g-cat"><i style="background:${color}"></i>${txt(category, 'Unknown')}</span>
    <p class="g-caption">${caption}</p>
  </div>`;
}

function animateGauge(score) {
  const arc = $('#gArc'), numEl = $('#gNum');
  if (!arc || !numEl || !isNum(score)) return;
  const s = Math.max(0, Math.min(100, score));
  const C = 2 * Math.PI * 52;
  // two frames so the browser paints the empty ring before the transition starts
  requestAnimationFrame(() => requestAnimationFrame(() => (arc.style.strokeDashoffset = (C * (1 - s / 100)).toFixed(2))));
  if (REDUCED_MOTION) { numEl.textContent = Math.round(s); return; }
  const t0 = performance.now();
  const tick = t => {
    const k = Math.min(1, (t - t0) / 1300);
    numEl.textContent = Math.round(s * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const chipsRow = chips => `<div class="chips">${chips.map(([k, v, s, tone]) => `<div class="chip ${tone ? 'tone-' + tone : ''}">
  <span class="chip-k">${k}</span><span class="chip-v">${txt(v)}</span>${s ? `<span class="chip-s">${esc(s)}</span>` : ''}</div>`).join('')}</div>`;

// ---------- results: leaf mode ----------
function renderResults(d) {
  showResults([
    printHead(d, 'Crop Health Report'),
    summaryCard(d),
    `<div class="grid-2">
      ${diagnosisCard(d.diagnosis)}
      ${cropCard(d.crop)}${diseaseCard(d.disease, d.diagnosis)}
      ${nutrientCard(d.nutrient)}${pestCard(d.pest)}
      ${adviceCard(d.recommendation)}
      ${explainCard(d.explain, false)}${breakdownCard(d.health)}
      ${card('', 'Photo Quality Checks', qualityBody(d.quality))}${runCard(d)}
    </div>`,
    resultFooter(d, 'Analyze another photo'),
  ].join(''), obj(d.health).score);
}

function summaryCard(d) {
  const h = obj(d.health), c = obj(d.crop), dg = obj(d.diagnosis), n = obj(d.nutrient), p = obj(d.pest), r = obj(d.recommendation);
  const prim = obj(n.primary), f0 = obj(arr(p.findings)[0]);
  const deficient = n.status === 'deficiency_detected';
  const color = /^#[0-9a-f]{3,8}$/i.test(h.color || '') ? h.color : HEALTH_FALLBACK[h.category] || '#1d7f47';
  const dgTone = { healthy: 'ok', disease: 'bad', pest: 'warn', deficiency: 'warn', abiotic: 'info' }[dg.kind] || '';
  const chips = [
    ['Crop', c.name, isNum(c.confidence) ? `${pct(c.confidence)} confidence` : '', 'info'],
    ['Diagnosis', dg.name, [dg.kind && kindInfo(dg.kind)[0], isNum(dg.confidence) && pct(dg.confidence)].filter(Boolean).join(' · '), dgTone],
    ['Nutrient', deficient ? prim.nutrient || 'Deficiency' : n.status ? 'Normal' : null, deficient && prim.level ? `${prim.level} likelihood` : '', deficient ? 'warn' : n.status ? 'ok' : ''],
    ['Pest', p.detected ? f0.name || 'Detected' : p.risk ? 'None detected' : null, p.risk ? riskText(p.risk) : '', p.detected ? TONE[p.risk] || 'warn' : p.risk ? 'ok' : ''],
    ['Urgency', r.urgency, r.urgency ? 'Action timing' : '', TONE[r.urgency] || ''],
  ];
  return `<section class="card" aria-label="Summary">
    <div class="summary">
      ${gaugeBox(h.score, color, 'out of 100', `Crop health score ${isNum(h.score) ? h.score : 'unknown'} out of 100${h.category ? ', ' + h.category : ''}`, h.category, 'Overall crop health score')}
      <div class="summary-main">
        <h2>${txt(c.name, 'Leaf')} · ${txt(dg.name, 'Unknown condition')}</h2>
        <p class="muted">${txt(r.summary, 'Analysis complete. See the detailed findings below.')}</p>
        ${chipsRow(chips)}
      </div>
    </div>
    ${warningsBanner(d)}
  </section>`;
}

const typeBadge = t => (t && !['None', 'Nutrient', 'Abiotic'].includes(t) ? badge(t, TYPE_TONE[t]) : '');
const condLabel = t => `${t.crop && t.crop !== '-' ? t.crop + ' · ' : ''}${t.name || t.key || '—'}`;

function diagnosisCard(dg) {
  dg = obj(dg);
  const top = arr(dg.top_k).map(obj);
  const gate = obj(obj(state.health).gates).domain_shift;
  const kp = obj(dg.kind_probabilities);
  return card('', 'Diagnosis &mdash; primary finding', `
    <div class="title-row"><div><p class="big">${txt(dg.name)}</p>${dg.crop ? `<p class="subtitle">on ${esc(dg.crop)}</p>` : ''}</div>
      <div class="badges">${dg.kind ? kindBadge(dg.kind) : ''}${typeBadge(dg.type)}</div></div>
    ${dg.low_confidence ? '<p class="inline-warn">Treat this result as indicative: the model is not confident, or the photo looks unlike its training images. Retake the photo in daylight with a single leaf filling the frame.</p>' : ''}
    <div class="pair">
      <div>
        ${meter('Model confidence', dg.confidence, confTone(dg.confidence))}
        ${!blank(dg.cause) ? `<p class="small"><strong>Cause:</strong> ${esc(dg.cause)}</p>` : ''}
        ${dg.description ? `<p>${esc(dg.description)}</p>` : ''}
        ${arr(dg.symptoms).length ? `<h4>Symptoms</h4>${bullets(dg.symptoms)}` : ''}
        ${!blank(dg.spread) ? `<p class="small"><strong>How it spreads:</strong> ${esc(dg.spread)}</p>` : ''}
      </div>
      <div>
        ${Object.keys(kp).length ? `<h4>What kind of problem is it?</h4>
          ${stackBar(KIND_ORDER.map(k => ({ label: kindInfo(k)[0], color: kindInfo(k)[2], v: kp[k] })), v => pct(v))}
          <p class="meter-note">Summed probability of all classes of each kind.</p>` : ''}
        ${top.length ? `<h4>Top-3 predictions</h4>${top.map(t => meterHtml(`${esc(condLabel(t))} ${kindBadge(t.kind)}`, t.confidence, 'b-info')).join('')}` : ''}
        ${isNum(dg.domain_shift) ? `<p class="small muted">Domain-shift score ${fix(dg.domain_shift, 2)}${isNum(gate) ? ` (flagged above ${fix(gate, 2)})` : ''} &mdash; how different this photo looks from the training images.</p>` : ''}
      </div>
    </div>`, 'span-2');
}

function cropCard(c) {
  c = obj(c);
  const alts = arr(c.alternatives).map(obj).filter(a => a.name);
  return card('1', 'Crop Variety Identification', `
    <p class="eyebrow-sm">Identified crop species</p>
    <p class="big">${txt(c.name)}${c.scientific_name ? ` <i class="sci">${esc(c.scientific_name)}</i>` : ''}</p>
    ${c.family ? `<p class="subtitle">Family ${esc(c.family)}</p>` : ''}
    ${meter('Identification confidence', c.confidence, confTone(c.confidence), pct(c.confidence), 'Sum of the probabilities of all classes of this crop')}
    ${c.description ? `<p>${esc(c.description)}</p>` : ''}
    ${arr(c.common_varieties).length ? `<h4>Popular varieties of this crop <span class="muted">(reference list &mdash; the model identifies the species, not the cultivar)</span></h4>${tags(c.common_varieties)}` : ''}
    ${alts.length ? `<h4>Other candidate crops</h4>${alts.map(a => meter(a.name, a.confidence, 'b-info', pct(a.confidence, 2))).join('')}` : ''}`);
}

function diseaseCard(dz, dg) {
  dz = obj(dz);
  const detected = dz.detected === true;
  const area = meter('Affected leaf area (image analysis)', isNum(dz.affected_area_pct) ? dz.affected_area_pct / 100 : null, 'b-bad', pctRaw(dz.affected_area_pct));
  return card('2.1', 'Disease Detection', `
    <div class="title-row"><p class="big">${txt(dz.name, detected ? 'Disease detected' : 'No infectious disease detected')}</p>
      ${detected ? typeBadge(dz.type) || badge('Disease', 'bad') : badge('Not detected', 'ok')}</div>
    ${detected && !blank(dz.pathogen) ? `<p class="subtitle">Pathogen: <i>${esc(dz.pathogen)}</i></p>` : ''}
    ${detected && dz.primary === false ? `<p class="inline-warn">Secondary finding: the primary diagnosis is ${txt(obj(dg).name)}, but this disease also has a notable probability &mdash; check the leaf for its symptoms.</p>` : ''}
    <div class="pair">
      ${detected ? meter('Confidence', dz.confidence, confTone(dz.confidence))
        : meter('Probability of any disease', dz.confidence, 'b-ok', pct(dz.confidence), 'Summed over all disease classes')}
      ${area}
    </div>
    ${dz.description ? `<p>${esc(dz.description)}</p>` : ''}
    ${detected && arr(dz.symptoms).length ? `<h4>Symptoms</h4>${bullets(dz.symptoms)}` : ''}
    ${detected && !blank(dz.spread) ? `<p class="small"><strong>How it spreads:</strong> ${esc(dz.spread)}</p>` : ''}`);
}

function nutrientCard(n) {
  n = obj(n);
  const deficient = n.status === 'deficiency_detected';
  const p = obj(n.primary);
  const inds = arr(n.indicators).map(obj);
  const learned = arr(n.learned).map(obj);
  const [srcLabel, srcTone] = NUTRIENT_SOURCE[p.source] || [];
  const symptoms = typeof p.symptoms === 'string' ? p.symptoms.split('; ').filter(Boolean) : [];
  return card('2.2', 'Nutrient Deficiency', `
    <div class="title-row"><p class="big">${deficient ? (p.nutrient ? `${esc(p.nutrient)} deficiency` : 'Deficiency detected') : n.status ? 'No deficiency detected' : '—'}</p>
      ${n.status ? badge(deficient ? 'Deficiency detected' : 'Normal', deficient ? 'warn' : 'ok') : ''}</div>
    ${deficient && n.primary ? `<div class="callout callout-warn">
        <div class="title-row"><strong>${txt(p.nutrient)}${p.symbol ? ` (${esc(p.symbol)})` : ''} · ${p.source === 'model' ? 'probability' : 'score'} ${fix(p.score, 2)}</strong>
          <span class="badges">${srcLabel ? badge(srcLabel, srcTone) : ''}${p.level ? badge(`${p.level} likelihood`, TONE[p.level]) : ''}</span></div>
        ${symptoms.length > 1 ? bullets(symptoms) : symptoms.length ? `<p>${esc(symptoms[0])}</p>` : ''}
        ${p.evidence ? `<p class="small"><strong>Evidence:</strong> ${esc(p.evidence)}</p>` : ''}</div>`
      : n.status ? '<p class="subtitle">Neither the learned deficiency classes nor the leaf-colour rules found a deficiency pattern.</p>' : ''}
    ${learned.length ? `<h4>Learned deficiency classes ${badge('Learned', 'info')}</h4>
      ${learned.map(l => meter(`${l.crop ? l.crop + ' · ' : ''}${l.name || l.key || '—'}`, l.probability, 'b-warn')).join('')}` : ''}
    ${inds.length ? `<h4>Colour-rule indicators <span class="muted">(ticks: Moderate 0.45 · High 0.70)</span></h4>
      ${inds.map(i => meter(`${i.nutrient || '—'}${i.symbol ? ` (${i.symbol})` : ''}`, i.score, `ticks ${levelBar(i.level)}`,
        `${fix(i.score, 2)}${i.level ? ' · ' + i.level : ''}`, i.evidence)).join('')}` : ''}
    ${profileBar(n.color_profile)}
    ${n.note ? `<p class="note">${esc(n.note)}</p>` : ''}`);
}

function profileBar(cp) {
  cp = obj(cp);
  const bars = stackBar(PROFILE.map(([k, label, color]) => ({ label, color, v: cp[k] })), v => `${v.toFixed(1)}%`);
  return bars ? `<h4>Leaf colour profile (% of leaf area)</h4>${bars}` : '';
}

function pestCard(p) {
  p = obj(p);
  const findings = arr(p.findings).map(obj);
  const title = p.detected ? 'Pest activity detected' : findings.length ? 'No confirmed pests' : 'No pests detected';
  return card('3.1', 'Pest Detection', `
    <div class="title-row"><p class="big">${title}</p>${p.risk ? badge(riskText(p.risk), TONE[p.risk]) : ''}</div>
    ${findings.length ? findings.map(f => {
      const [label, tone] = PEST_SOURCE[f.source] || [f.source || 'Finding', 'neutral'];
      return `<div class="finding">
        <div class="title-row"><div><strong>${txt(f.name)}</strong>${f.scientific_name ? ` <i class="muted">${esc(f.scientific_name)}</i>` : ''}</div>${badge(label, tone)}</div>
        ${meter('Confidence', f.confidence, 'b-info')}
        ${f.evidence ? `<p class="small">${esc(f.evidence)}</p>` : ''}
        ${arr(f.control).length ? `<h5>Control measures</h5>${bullets(f.control)}` : ''}</div>`;
    }).join('') : '<p class="empty-ok">No learned pest-damage class, chewing holes, mite stippling or vector-borne disease signs on this leaf.</p>'}
    ${p.note ? `<p class="note">${esc(p.note)}</p>` : ''}`);
}

function adviceCard(r) {
  r = obj(r);
  const fert = arr(r.fertilizers).map(obj), treat = arr(r.treatment).map(obj), s = obj(r.npk_schedule);
  const cols = [['Fertilizer', 'name'], ['NPK', 'npk'], ['Dose', 'dose'], ['Method', 'method'], ['Timing', 'timing'], ['Why', 'reason']];
  const listBlock = (title, items, tone) => (arr(items).length ? `<div class="list-block tone-${tone}"><h4>${title}</h4>${bullets(items)}</div>` : '');
  const hasSchedule = s.crop || isNum(s.N) || isNum(s.P2O5) || isNum(s.K2O);
  return card('3.2', 'Fertilizer &amp; Treatment', `
    <div class="title-row"><p style="max-width:80ch;margin:0">${txt(r.summary, 'No recommendation available.')}</p>${r.urgency ? badge(`Urgency: ${r.urgency}`, TONE[r.urgency]) : ''}</div>
    ${fert.length ? `<h4>Fertilizer recommendation</h4>
      <div class="table-wrap"><table class="table stack-sm"><thead><tr>${cols.map(([h]) => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>
      ${fert.map(f => `<tr>${cols.map(([h, k], j) => `<td data-label="${h}">${j === 0 ? `<strong>${txt(f[k])}</strong>` : txt(f[k])}</td>`).join('')}</tr>`).join('')}
      </tbody></table></div>` : ''}
    ${treat.length ? `<h4>Disease, pest &amp; nutrient treatment</h4><ul class="treat">${treat.map(t => `<li>
        <div class="title-row"><strong>${txt(t.name)}</strong>${t.type ? badge(t.type, TREAT_TONE[t.type]) : ''}</div>
        <p class="small">${!blank(t.dose) ? `<strong>Dose:</strong> ${esc(t.dose)}` : ''}${!blank(t.dose) && t.notes ? ' &mdash; ' : ''}${esc(t.notes || '')}</p></li>`).join('')}</ul>` : ''}
    ${arr(r.organic).length || arr(r.cultural).length || arr(r.avoid).length ? `<div class="cols-3">${listBlock('Organic options', r.organic, 'ok')}${listBlock('Cultural practices', r.cultural, 'info')}${listBlock('Avoid', r.avoid, 'bad')}</div>` : ''}
    ${hasSchedule ? `<h4>Recommended NPK dose${s.crop ? ' for ' + esc(s.crop) : ''}${s.unit ? ` <span class="muted">(${esc(s.unit)})</span>` : ''}</h4>
      <div class="npk">${[['N', s.N], ['P₂O₅', s.P2O5], ['K₂O', s.K2O]].map(([k, v]) =>
        `<div><span>${k}</span><strong>${isNum(v) ? v : '—'}</strong><small>${txt(s.unit, 'kg/ha')}</small></div>`).join('')}</div>
      ${s.split ? `<p class="small" style="margin-top:10px"><strong>Split:</strong> ${esc(s.split)}</p>` : ''}${s.note ? `<p class="small muted">${esc(s.note)}</p>` : ''}` : ''}
    ${safetyNote(r.safety)}`, 'span-2');
}

const safetyNote = s => (s ? `<div class="callout callout-safety" role="note"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 4 5v6c0 5 3.4 9.3 8 11 4.6-1.7 8-6 8-11V5l-8-3zm-1 14-4-4 1.4-1.4L11 13.2l4.6-4.6L17 10l-6 6z" fill="currentColor"/></svg>
  <div><strong>Safety:</strong> ${esc(s)}</div></div>` : '');

function explainCard(ex, insect) {
  ex = obj(ex);
  const views = [
    ['orig', 'Original', state.url || '', 'The photo you uploaded.'],
    ['cam', 'Grad-CAM', safeUrl(ex.gradcam), `Grad-CAM heat-map: warm colours mark the regions that most influenced the ${insect ? 'insect' : 'leaf'} model.`],
    ['seg', 'Segmentation', safeUrl(ex.segmentation), 'Leaf mask and symptom map (background dimmed):', SEG_LEGEND],
  ].filter(v => v[2]);
  if (!views.length) return '';
  return card('', 'Visual Explanation', `
    <div class="tabs" role="tablist" aria-label="Visual explanation views">${views.map(([id, label], k) =>
      `<button type="button" role="tab" id="tab-${id}" aria-controls="panel-${id}" aria-selected="${k === 0}" tabindex="${k === 0 ? 0 : -1}">${label}</button>`).join('')}</div>
    <div class="xpanels">${views.map(([id, label, src, caption, legend], k) =>
      `<div class="xpanel" role="tabpanel" id="panel-${id}" aria-labelledby="tab-${id}"${k === 0 ? '' : ' hidden'}>
        <figure><img src="${esc(src)}" alt="${esc(label)} view of the analysed photo"><figcaption>${esc(caption)}${legend || ''}</figcaption></figure></div>`).join('')}</div>
    ${safeUrl(ex.gradcam) ? '' : '<p class="small muted">A Grad-CAM heat-map is not available for this result.</p>'}`);
}

function selectTab(tab) {
  const list = tab.closest('[role="tablist"]');
  $$('[role="tab"]', list).forEach(t => {
    const on = t === tab;
    t.setAttribute('aria-selected', on);
    t.tabIndex = on ? 0 : -1;
    $('#' + t.getAttribute('aria-controls')).hidden = !on;
  });
  tab.focus();
}

function breakdownCard(h) {
  h = obj(h);
  const rows = arr(h.breakdown).map(obj);
  const max = Math.max(40, ...rows.map(r => (isNum(r.penalty) ? r.penalty : 0)));
  const total = rows.reduce((s, r) => s + (isNum(r.penalty) ? r.penalty : 0), 0);
  const capped = isNum(h.score) && h.score < Math.round(100 - total);
  return card('', 'Health Score Breakdown', `
    <p class="small muted">Transparent penalty model: the score starts at 100 and each factor subtracts points.</p>
    ${rows.map(r => meter(r.factor || '—', isNum(r.penalty) ? r.penalty / max : 0, 'b-bad', r.penalty > 0 ? `−${fix(r.penalty)} pts` : '0 pts', r.detail)).join('')}
    ${rows.length ? `<div class="equation">100 − ${fix(total)} penalty = <strong>${isNum(h.score) ? h.score : '—'}</strong> ${h.category ? `(${esc(h.category)})` : ''}
      ${capped ? '<div class="small muted">Score capped because a problem was diagnosed with good confidence.</div>' : ''}</div>` : '<p class="muted">No breakdown available.</p>'}`);
}

function qualityBody(q) {
  q = obj(q);
  return `<div class="title-row"><p style="margin:0">Photo quality score <strong>${isNum(q.score) ? q.score : '—'}/100</strong></p>
      ${q.passed === false ? badge('Failed — forced', 'bad') : q.passed === true ? badge('Passed', 'ok') : ''}</div>
    ${checksTable(arr(q.checks))}`;
}

function runCard(d) {
  const t = obj(d.timing_ms), m = obj(d.model), im = obj(obj(d.insect).model);
  const stages = [['quality', 'Quality gate'], ['classify', 'Leaf model + Grad-CAM'], ['pest_model', 'Insect model'],
    ['analysis', 'Leaf image analysis'], ['advisory', 'Advisory engine']].filter(([k]) => isNum(t[k]));
  const total = isNum(t.total) ? t.total : stages.reduce((s, [k]) => s + t[k], 0);
  const classes = [isNum(m.classes) && `${m.classes} classes`, isNum(m.crops) && `${m.crops} crops`].filter(Boolean).join(' · ');
  return card('', 'Run Details', `
    <dl class="kv">
      <dt>Leaf model</dt><dd>${txt(m.name)}${m.arch ? ` <code>${esc(m.arch)}</code>` : ''}</dd>
      ${classes ? `<dt>Classes</dt><dd>${esc(classes)}</dd>` : ''}
      ${im.name ? `<dt>Insect model</dt><dd>${esc(im.name)}${isNum(im.species) ? ` · ${im.species} species` : ''}</dd>` : ''}
      <dt>Runs on</dt><dd>${txt(state.device, 'CPU')} · offline</dd>
      <dt>Request ID</dt><dd><code>${txt(d.request_id)}</code></dd>
    </dl>
    <h4>Processing time</h4>
    ${stages.map(([k, label]) => meter(label, total ? t[k] / total : 0, 'b-info', `${fix(t[k])} ms`)).join('')}
    <p class="equation">Total <strong>${fix(total)} ms</strong></p>`);
}

// ---------- results: insect mode ----------
function renderInsect(d) {
  const ins = obj(d.insect), sp = obj(ins.species), cands = arr(ins.candidates).map(obj);
  const conf = isNum(sp.confidence) ? Math.round(sp.confidence * 100) : null;
  const color = { High: '#2f9e44', Moderate: '#e0a82e', Low: '#c53030' }[confLevel(sp.confidence)] || '#1d7f47';
  const control = arr(sp.control);
  showResults([
    printHead(d, 'Insect Pest Report'),
    `<section class="card" aria-label="Summary">
      <div class="summary">
        ${gaugeBox(conf, color, '% sure', `Species confidence ${conf ?? 'unknown'} percent`, confLevel(sp.confidence) && `${confLevel(sp.confidence)} confidence`, 'Species confidence')}
        <div class="summary-main">
          <p class="eyebrow-sm">Insect pest identified</p>
          <h2>${txt(sp.name, 'Unknown insect')}</h2>
          ${sp.scientific_name ? `<p class="sci-line"><i>${esc(sp.scientific_name)}</i></p>` : ''}
          ${chipsRow([
            ['Mode', 'Insect photo', isNum(ins.router_confidence) ? `Routed with ${pct(ins.router_confidence)} confidence` : '', 'info'],
            ['Species', sp.name, isNum(sp.confidence) ? `${pct(sp.confidence)} confidence` : '', 'warn'],
            ['Host crops', arr(sp.crops).length || null, arr(sp.crops).slice(0, 3).join(', '), ''],
            ['Control steps', control.length || null, control.length ? 'See the checklist below' : '', 'ok'],
          ])}
        </div>
      </div>
      ${warningsBanner(d)}
    </section>`,
    `<div class="grid-2">
      ${card('3.1', 'Top-3 candidate species', `
        <p class="small muted">Many pests look alike at phone-camera resolution &mdash; compare the photo with the candidates below.</p>
        ${cands.map((c, k) => `<div class="candidate${k === 0 ? ' is-top' : ''}">
          ${meterHtml(`<strong>${txt(c.name)}</strong>${c.scientific_name ? ` <i class="muted">${esc(c.scientific_name)}</i>` : ''}`, c.confidence, k === 0 ? confTone(c.confidence) : 'b-info')}
          ${arr(c.crops).length ? `<p class="meter-note">Attacks: ${esc(arr(c.crops).join(', '))}</p>` : ''}</div>`).join('')}
        ${isNum(ins.router_confidence) ? `<p class="note">The leaf model first routed this photo as an insect (${pct(ins.router_confidence)} sure); the insect model${obj(ins.model).name ? ` (${esc(obj(ins.model).name)}${isNum(obj(ins.model).species) ? `, ${obj(ins.model).species} species` : ''})` : ''} then named the species.</p>` : ''}`)}
      ${card('', 'Host crops &amp; damage', `
        ${arr(sp.crops).length ? `<h4>Crops it attacks</h4>${tags(sp.crops)}` : ''}
        ${sp.damage ? `<h4>Damage</h4><p>${esc(sp.damage)}</p>` : ''}
        ${!arr(sp.crops).length && !sp.damage ? '<p class="muted">No host-crop or damage details for this species in the knowledge base.</p>' : ''}`)}
      ${card('3.2', 'Control checklist', control.length ? `
        <p class="small muted">Work through these in order &mdash; cultural and biological steps first, chemicals only above the action threshold.</p>
        <ul class="checklist">${control.map((c, k) => `<li><label><input type="checkbox" id="ctl-${k}"><span>${esc(c)}</span></label></li>`).join('')}</ul>
        ${safetyNote(ins.safety)}`
        : '<p class="muted">No control measures in the knowledge base for this species &mdash; ask your local KVK.</p>', 'span-2')}
      ${explainCard(d.explain, true)}${card('', 'Photo Quality Checks', qualityBody(d.quality))}
      ${runCard(d)}
    </div>`,
    resultFooter(d, 'Analyze another photo'),
  ].join(''), conf);
}

// ---------- sortable / filterable tables ----------
// state.tables[id] = { rows, cols: [[key, label, cls, render(row)]], sort: {key, dir}, filter?() -> row => bool, onRender?(n) }
const cmp = (a, b) => (typeof a === 'string' || typeof b === 'string'
  ? String(a ?? '').localeCompare(String(b ?? '')) : (isNum(a) ? a : -1) - (isNum(b) ? b : -1));

function tableShell(id, cols) {
  return `<div class="table-wrap scroll"><table class="table" id="${id}"><thead><tr>${cols.map(([k, label, cls]) =>
    `<th scope="col" class="${cls}" data-key="${k}"><button type="button" class="sort-btn" data-sort="${k}" data-table="${id}">${label}</button></th>`).join('')}</tr></thead>
    <tbody></tbody></table></div>`;
}

function renderTable(id) {
  const t = state.tables[id], table = document.getElementById(id);
  if (!t || !table) return;
  const { key, dir } = t.sort;
  let rows = t.filter ? t.rows.filter(t.filter()) : t.rows;
  if (key) rows = rows.slice().sort((a, b) => cmp(a[key], b[key]) * dir);
  $$('th', table).forEach(th => (th.dataset.key === key ? th.setAttribute('aria-sort', dir > 0 ? 'ascending' : 'descending') : th.removeAttribute('aria-sort')));
  $('tbody', table).innerHTML = rows.length
    ? rows.map(r => `<tr>${t.cols.map(([, , cls, render]) => `<td class="${cls}">${render(r)}</td>`).join('')}</tr>`).join('')
    : `<tr><td colspan="${t.cols.length}" class="muted">No rows match the filters.</td></tr>`;
  if (t.onRender) t.onRender(rows.length);
}

function sortTable(id, key) {
  const t = state.tables[id];
  if (!t) return;
  const text = t.rows.length && typeof t.rows[0][key] === 'string';
  t.sort = { key, dir: t.sort.key === key ? -t.sort.dir : text ? 1 : -1 }; // text A→Z first, numbers high→low first
  renderTable(id);
}

const f1Cell = v => `${pct(v)}<span class="cell-bar ${isNum(v) && v < 0.9 ? 'low' : ''}" aria-hidden="true"><span style="width:${(clamp01(v) * 100).toFixed(0)}%"></span></span>`;

// ---------- model performance ----------
async function loadMetrics(background) {
  const box = $('#perfBody');
  state.metricsAt = Date.now();
  let m;
  try {
    m = obj(await api('/api/metrics'));
  } catch (e) {
    if (background === true && state.metricsSig) return; // silent refresh failed: keep what is shown
    state.metricsSig = null;
    box.innerHTML = `<div class="card empty"><span>Model metrics are unavailable right now (${esc(e.message)}).</span>
      <button type="button" class="btn btn-ghost" data-action="metrics">Retry</button></div>`;
    return;
  }
  const models = arr(m.leaf_models).map(obj);
  clearTimeout(state.metricsTimer);
  if (!models.length) state.metricsTimer = setTimeout(loadMetrics, 60000); // pick up results once training finishes
  const sig = JSON.stringify(m);
  if (sig === state.metricsSig) return; // unchanged: keep the DOM as is
  state.metricsSig = sig;

  const deployed = obj(m.deployed), leaf = obj(m.leaf), ds = obj(m.dataset), report = obj(m.report);
  const primary = models.find(x => x.arch === deployed.leaf) || models[0];
  const perClass = arr(leaf.per_class).map(obj).map(r => ({ ...r, label: r.label || r.key, crop: splitLabel(r.label)[0] || String(r.key || '').split('___')[0] }));
  const router = perClass.filter(r => r.kind === 'other'); // "not a plant" / "insect" routing classes
  const classes = perClass.filter(r => r.kind !== 'other');
  const routerCrops = new Set(router.map(r => String(r.key).split('___')[0]));
  const labels = Object.fromEntries(perClass.map(r => [r.key, r.label]));

  $('#perfLead').textContent = perfLead(primary, ds);
  box.innerHTML = `<div class="perf">
    ${primary ? kpiBlock(primary) : trainingBlock()}
    ${primary ? byKindBlock(primary, ds, router) : ''}
    ${primary ? `<div class="grid-2 perf-pair">${imageryBlock(primary)}${calibrationBlock(obj(obj(m.calibration).leaf), report.selective_test)}</div>` : ''}
    ${models.length ? comparisonBlock(models, primary, obj(report.leaf_val_f1)) : ''}
    ${pestBlock(m)}
    ${robustnessBlock(report.robustness)}
    ${figuresBlock(m)}
    <div class="grid-2 perf-pair">${datasetBlock(ds)}${confusionsBlock(leaf.top_confusions, labels)}</div>
    ${perCropBlock(leaf.by_crop, ds.by_crop, routerCrops)}
    ${perClassBlock(classes)}
  </div>`;
  renderTable('perCrop');
  renderTable('perClass');
}

function perfLead(p, ds) {
  if (!p) return 'Metrics are computed on images the models never saw during training or model selection.';
  const data = [isNum(ds.leaf_images) && `${num(ds.leaf_images)} photos`, isNum(ds.sources) && `from ${ds.sources} public datasets`].filter(Boolean).join(' ');
  const scope = [isNum(p.classes) && `${p.classes} classes`, isNum(ds.crops) && `${ds.crops} crops`].filter(Boolean).join(', ');
  return `Deployed leaf model: ${p.name || p.arch || 'CNN'}${data ? `, trained on ${data}` : ''}${scope ? ` (${scope})` : ''}. `
    + `Every number below is measured on the ${isNum(p.n) ? num(p.n) + ' ' : ''}held-out test images, never used for training or model selection.`;
}

function trainingBlock() {
  return `<div class="card empty"><span class="spinner" aria-hidden="true"></span>
    <div style="flex:1;min-width:220px"><h3 style="margin:0 0 4px">Training in progress</h3>
    <p class="muted" style="margin:0">The models are still being trained and evaluated. Accuracy, precision, recall, F1-score and the figures will appear here automatically when evaluation finishes.</p></div>
    <button type="button" class="btn btn-ghost" data-action="metrics">Refresh now</button></div>`;
}

const kpiTiles = tiles => `<div class="kpis">${tiles.map(([k, v, s]) =>
  `<div class="kpi"><div class="kpi-k">${k}</div><div class="kpi-v">${esc(v)}</div><div class="kpi-s">${esc(s || '')}</div></div>`).join('')}</div>`;
const latencyTile = lat => ['CPU latency (p95)', isNum(lat.p95) ? `${fix(lat.p95, 1)} ms` : '—',
  isNum(lat.p50) ? `p50 ${fix(lat.p50, 1)} ms${isNum(lat.p99) ? ` · p99 ${fix(lat.p99, 1)} ms` : ''}` : 'Per image, batch size 1'];

function kpiBlock(p) {
  const meta = [
    ['Test images', isNum(p.n) ? num(p.n) : null], ['Classes', isNum(p.classes) ? p.classes : null],
    ['Weighted F1', isNum(p.f1_weighted) ? pct(p.f1_weighted, 2) : null], ['Parameters', isNum(p.params_m) ? `${fix(p.params_m, 2)} M` : null],
    ['Model size', isNum(p.size_mb) ? `${fix(p.size_mb, 1)} MB` : null],
    ['Best epoch', isNum(p.best_epoch) ? `${p.best_epoch}${isNum(p.epochs) ? ' / ' + p.epochs : ''}` : null],
    ['Training time', isNum(p.train_time_min) ? `${fix(p.train_time_min, 1)} min` : null],
  ].filter(([, v]) => v != null);
  return `<div><h3 class="block-title">Deployed leaf model <span>${txt(p.name || p.arch)}</span></h3>
    ${kpiTiles([
      ['Accuracy', pct(p.accuracy, 2), ci(p.accuracy_ci95)],
      ['F1-score (macro)', pct(p.f1_macro, 2), ci(p.f1_macro_ci95)],
      ['Precision (macro)', pct(p.precision_macro, 2), 'Unweighted mean over classes'],
      ['Recall (macro)', pct(p.recall_macro, 2), 'Unweighted mean over classes'],
      ['Top-3 accuracy', pct(p.top3_accuracy, 2), 'Right answer among the top 3'],
      ['Crop identification', pct(p.crop_accuracy, 2), 'Right crop, any condition'],
      ['ROC-AUC', fix(p.roc_auc_macro_ovr, 4), 'Macro, one-vs-rest'],
      latencyTile(obj(p.latency_ms_cpu)),
    ])}
    <div class="meta-row">${meta.map(([k, v]) => `<span>${k}: <b>${esc(v)}</b></span>`).join('')}</div></div>`;
}

function byKindBlock(p, ds, router) {
  const bk = obj(p.by_kind), dk = obj(ds.by_kind);
  const rows = KIND_ORDER.filter(k => Object.keys(obj(bk[k])).length).map(k => [k, obj(bk[k]), obj(dk[k])]);
  if (!rows.length) return '';
  return `<div class="card"><h3 class="card-title">Accuracy by kind of finding</h3>
    <div class="table-wrap"><table class="table"><thead><tr><th scope="col">Kind</th><th scope="col" class="num">Classes</th><th scope="col" class="num">Test images</th>
      <th scope="col" class="num">Accuracy</th><th scope="col" class="num">Precision</th><th scope="col" class="num">Recall</th><th scope="col" class="num">F1 (macro)</th></tr></thead>
    <tbody>${rows.map(([k, r, d]) => `<tr><td>${kindBadge(k)}</td><td class="num">${num(d.classes)}</td><td class="num">${num(r.n)}</td>
      <td class="num">${pct(r.accuracy)}</td><td class="num">${pct(r.precision_macro)}</td><td class="num">${pct(r.recall_macro)}</td>
      <td class="num">${f1Cell(r.f1_macro)}</td></tr>`).join('')}</tbody></table></div>
    ${router.length ? `<p class="small muted" style="margin:10px 0 0">Photo router classes: ${router.map(r =>
      `${esc(r.label)} recall ${pct(r.recall)} (${num(r.support)} test photos)`).join(' · ')}.</p>` : ''}</div>`;
}

function imageryBlock(p) {
  const bi = obj(p.by_imagery);
  const rows = [['lab', 'Lab photos', 'plain background'], ['field', 'Field photos', 'real farm conditions'], ['mixed', 'Mixed sources', 'lab and field']]
    .map(([k, label, note]) => [label, note, obj(bi[k])]).filter(([, , r]) => isNum(r.accuracy));
  if (!rows.length) return '';
  const lab = obj(bi.lab), field = obj(bi.field);
  const gap = isNum(lab.accuracy) && isNum(field.accuracy) ? (lab.accuracy - field.accuracy) * 100 : null;
  return `<div class="card"><h3 class="card-title">Lab vs field photos</h3>
    ${rows.map(([label, note, r]) => meter(`${label} · ${note}`, r.accuracy, confTone(r.accuracy), `${pct(r.accuracy)} accuracy`,
      `${num(r.n)} test images · macro F1 ${pct(r.f1_macro)}`)).join('')}
    ${isNum(gap) && gap > 0 ? `<p class="callout callout-warn small">Field photos are ${fix(gap, 1)} percentage points harder than lab photos. A farmer&rsquo;s phone photo is a field photo, so ${pct(field.accuracy)} is the realistic accuracy to expect.</p>` : ''}</div>`;
}

function calibrationBlock(c, selective) {
  const sel = arr(selective).map(obj).filter(r => isNum(r.threshold));
  if (!isNum(c.ece) && !sel.length) return '';
  const gate = obj(obj(state.health).gates).low_confidence;
  const stats = [
    ['Calibration error (ECE)', isNum(c.ece) ? `${pct(c.ece)} → ${pct(c.ece_calibrated)}` : null, 'before → after temperature scaling'],
    ['Temperature', isNum(c.temperature) ? `T = ${fix(c.temperature, 2)}` : null, 'fitted on the validation split'],
    ['Validation NLL', isNum(c.val_nll) ? `${fix(c.val_nll, 3)} → ${fix(c.val_nll_calibrated, 3)}` : null, 'lower is better'],
  ].filter(([, v]) => v);
  return `<div class="card"><h3 class="card-title">Calibration &amp; confidence gate</h3>
    ${stats.length ? `<div class="mini-stats">${stats.map(([k, v, s]) => `<div class="mini-stat"><span class="k">${k}</span><strong>${esc(v)}</strong><span class="s">${s}</span></div>`).join('')}</div>` : ''}
    ${sel.length ? `<h4 class="sub-title">Selective prediction on the test set</h4>
      <div class="table-wrap"><table class="table"><thead><tr><th scope="col">Confidence ≥</th><th scope="col" class="num">Photos answered</th><th scope="col" class="num">Accuracy on those</th></tr></thead>
      <tbody>${sel.map(r => { const on = isNum(gate) && Math.abs(r.threshold - gate) < 1e-6; return `<tr class="${on ? 'is-primary' : ''}">
        <td>${r.threshold > 0 ? fix(r.threshold, 2) : 'No gate'}${on ? ' ' + badge('App gate', 'ok') : ''}</td>
        <td class="num">${pct(r.coverage)}</td><td class="num">${pct(r.accuracy)}</td></tr>`; }).join('')}</tbody></table></div>
      <p class="small muted" style="margin:8px 0 0">Below the app&rsquo;s gate a result is still shown, but flagged as low confidence.</p>` : ''}</div>`;
}

function comparisonBlock(models, primary, valF1) {
  const lat = m => obj(m.latency_ms_cpu).p95;
  const field = m => obj(obj(m.by_imagery).field).accuracy;
  return `<div class="card"><h3 class="card-title">Leaf model comparison</h3>
    <div class="table-wrap"><table class="table"><thead><tr><th scope="col">Model</th><th scope="col" class="num">Accuracy</th><th scope="col" class="num">F1 (macro)</th>
      <th scope="col" class="num">Top-3</th><th scope="col" class="num">Crop acc.</th><th scope="col" class="num">Field acc.</th><th scope="col" class="num">Val F1</th>
      <th scope="col" class="num">Params (M)</th><th scope="col" class="num">Size (MB)</th><th scope="col" class="num">CPU p95 (ms)</th><th scope="col" class="num">Train (min)</th></tr></thead>
    <tbody>${models.map(m => `<tr class="${m === primary ? 'is-primary' : ''}"><td>${txt(m.name || m.arch)}${m === primary ? ' ' + badge('Deployed', 'ok') : ''}</td>
      <td class="num">${pct(m.accuracy, 2)}</td><td class="num">${pct(m.f1_macro, 2)}</td><td class="num">${pct(m.top3_accuracy, 2)}</td>
      <td class="num">${pct(m.crop_accuracy, 2)}</td><td class="num">${pct(field(m), 2)}</td><td class="num">${pct(valF1[m.arch], 2)}</td>
      <td class="num">${fix(m.params_m, 2)}</td><td class="num">${fix(m.size_mb, 1)}</td><td class="num">${fix(lat(m), 1)}</td><td class="num">${fix(m.train_time_min, 0)}</td></tr>`).join('')}</tbody></table></div>
    ${models.length < 2 ? '<p class="small muted" style="margin:10px 0 0">Other architectures are added here when their evaluation finishes.</p>' : ''}</div>`;
}

function pestBlock(m) {
  const models = arr(m.pest_models).map(obj), ds = obj(m.dataset);
  const p = models.find(x => x.arch === obj(m.deployed).pest) || models[0];
  if (!p) {
    const prepared = [isNum(ds.pest_species) && `${ds.pest_species} species`, isNum(ds.pest_images) && `${num(ds.pest_images)} images`].filter(Boolean).join(', ');
    return `<div class="card empty pest-pending"><span class="badge badge-teal">Insect model</span>
      <p style="margin:0;flex:1;min-width:220px">The insect-pest identification model (IP102${prepared ? ` &mdash; ${prepared}` : ''}) has not been evaluated yet. Until it is deployed, insect photos fall back to leaf analysis with a warning.</p></div>`;
  }
  return `<div><h3 class="block-title">Insect-pest model <span>${txt(p.name || p.arch)}${isNum(p.classes) ? ` · ${p.classes} species` : ''}</span></h3>
    ${kpiTiles([
      ['Accuracy', pct(p.accuracy, 2), ci(p.accuracy_ci95)],
      ['F1-score (macro)', pct(p.f1_macro, 2), ci(p.f1_macro_ci95)],
      ['Top-3 accuracy', pct(p.top3_accuracy, 2), 'Right species among the top 3'],
      latencyTile(obj(p.latency_ms_cpu)),
    ])}
    ${models.length > 1 ? `<p class="small muted">${models.map(x => `${esc(x.name || x.arch)}: ${pct(x.accuracy, 2)} accuracy, F1 ${pct(x.f1_macro, 2)}`).join(' · ')}</p>` : ''}</div>`;
}

function robustnessBlock(rob) {
  const rows = arr(rob).map(obj);
  if (!rows.length) return '';
  const cols = [['accuracy', 'Accuracy'], ['f1_macro', 'F1 (macro)'], ['gate_pass_rate', 'Gate pass'], ['accuracy_after_gate', 'Acc. after gate']]
    .filter(([k]) => rows.some(r => isNum(r[k]))); // the report may not have every column
  return `<div class="card"><h3 class="card-title">Robustness to real-world photo conditions</h3>
    <p class="small muted">Test set after simulated field conditions (blur, noise, lighting, JPEG compression…). Gate pass = share of photos the quality gate accepts.</p>
    <div class="table-wrap"><table class="table"><thead><tr><th scope="col">Condition</th>${cols.map(([, l]) => `<th scope="col" class="num">${l}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r => `<tr><td>${txt(r.condition || r.name)}</td>${cols.map(([k]) => `<td class="num">${pct(r[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div>`;
}

const FIGURE_NAMES = {
  model_comparison: 'Model comparison', lab_vs_field: 'Lab vs field photos', by_kind: 'Accuracy by kind of finding',
  selective_prediction: 'Selective prediction (accuracy vs coverage)', robustness: 'Robustness to image corruptions',
  dataset_crops: 'Images per crop', training_curves: 'Training curves', per_crop_accuracy: 'Per-crop accuracy',
  per_class_f1_hist: 'Per-class F1 distribution',
};

function figuresBlock(m) {
  const named = (figs, prefix) => Object.entries(obj(figs)).map(([k, u]) => [prefix + (FIGURE_NAMES[k] || k.replace(/_/g, ' ')), safeUrl(u)]);
  const list = [...named(m.figures, ''), ...named(obj(m.leaf).figures, 'Leaf model: '), ...named(obj(m.pest).figures, 'Insect model: ')].filter(([, u]) => u);
  if (!list.length) return '';
  return `<div><h3 class="block-title">Evaluation figures <span>click to enlarge</span></h3><div class="figs">${list.map(([label, u]) =>
    `<figure class="fig"><button type="button" data-lightbox="${esc(u)}" data-caption="${esc(label)}" aria-label="Enlarge: ${esc(label)}">
      <img src="${esc(u)}" alt="${esc(label)}" loading="lazy"></button><figcaption>${esc(label)}</figcaption></figure>`).join('')}</div></div>`;
}

function datasetBlock(ds) {
  if (!Object.keys(ds).length) return '';
  const sp = obj(ds.split);
  const parts = [['Train', sp.train, '#1d7f47'], ['Validation', sp.val, '#b86a14'], ['Test', sp.test, '#1d5fa8']].filter(([, v]) => isNum(v) && v > 0);
  const sum = parts.reduce((s, [, v]) => s + v, 0);
  const bk = obj(ds.by_kind), bi = obj(ds.by_imagery);
  const kinds = [...KIND_ORDER, 'other'].map(k => ({ label: k === 'other' ? 'Router (non-plant / insect)' : kindInfo(k)[0], color: kindInfo(k)[2], v: obj(bk[k]).images }));
  const imagery = [['field', 'Field', '#1d7f47'], ['lab', 'Lab', '#1d5fa8'], ['mixed', 'Mixed', '#b86a14'], ['other', 'Router photos', '#a3aea8']]
    .map(([k, label, color]) => ({ label, color, v: bi[k] }));
  const pests = [isNum(ds.pest_images) && `${num(ds.pest_images)} images`, isNum(ds.pest_species) && `${ds.pest_species} species`].filter(Boolean).join(' · ');
  return `<div class="card"><h3 class="card-title">Dataset</h3>
    <dl class="kv"><dt>Leaf images</dt><dd>${num(ds.leaf_images)}</dd>
      <dt>Classes</dt><dd>${txt(ds.leaf_classes)}${isNum(ds.crops) ? ` across ${ds.crops} crops` : ''}</dd>
      ${isNum(ds.sources) ? `<dt>Sources</dt><dd>${ds.sources} public datasets</dd>` : ''}
      ${pests ? `<dt>Insect pests</dt><dd>${esc(pests)} (IP102)</dd>` : ''}</dl>
    ${sum ? `<h4 class="sub-title">Train / validation / test split</h4>
      <div class="split-bar" role="img" aria-label="${esc(parts.map(([k, v]) => `${k} ${num(v)}`).join(', '))}">${parts.map(([k, v, c]) =>
        `<span style="width:${((v / sum) * 100).toFixed(1)}%;background:${c}">${v / sum >= 0.2 ? `${k} ${((v / sum) * 100).toFixed(0)}%` : `${((v / sum) * 100).toFixed(0)}%`}</span>`).join('')}</div>
      <p class="small muted">${parts.map(([k, v]) => `${k}: ${num(v)}`).join(' · ')}</p>` : ''}
    ${ds.split_method ? `<p class="small">${esc(ds.split_method)}</p>` : ''}
    ${Object.keys(bk).length ? `<h4 class="sub-title">Images by kind</h4>${stackBar(kinds, num)}` : ''}
    ${Object.keys(bi).length ? `<h4 class="sub-title">Images by photo type</h4>${stackBar(imagery, num)}` : ''}</div>`;
}

function confusionsBlock(conf, labels) {
  const rows = arr(conf).map(obj).filter(r => r.true && r.pred).slice(0, 12);
  if (!rows.length) return '';
  const name = k => labels[k] || String(k).replace(/___/g, ' · ').replace(/_/g, ' ');
  return `<div class="card"><h3 class="card-title">Most confused pairs</h3>
    <p class="small muted">Test images of the first class predicted as the second &mdash; mostly look-alike symptoms on the same crop.</p>
    <ol class="confusions">${rows.map(r => `<li><span>${esc(name(r.true))}</span><span class="arrow" aria-label="predicted as">→</span><span>${esc(name(r.pred))}</span><b>${num(r.n)}</b></li>`).join('')}</ol></div>`;
}

function perCropBlock(byCrop, dsCrops, routerCrops) {
  const rows = Object.entries(obj(byCrop)).filter(([crop]) => !routerCrops.has(crop))
    .map(([crop, r]) => ({ crop, classes: obj(obj(dsCrops)[crop]).classes, ...obj(r) }));
  if (!rows.length) return '';
  const cols = [
    ['crop', 'Crop', '', r => `<strong>${txt(r.crop)}</strong>`],
    ['classes', 'Classes', 'num', r => num(r.classes)],
    ['n', 'Test images', 'num', r => num(r.n)],
    ['accuracy', 'Accuracy', 'num', r => f1Cell(r.accuracy)],
    ['f1_macro', 'F1 (macro)', 'num', r => pct(r.f1_macro)],
  ];
  state.tables.perCrop = { rows, cols, sort: { key: 'accuracy', dir: 1 } }; // weakest crops first
  return `<div class="card"><h3 class="card-title">Accuracy per crop <span class="muted small">${rows.length} crops · click a column to sort</span></h3>
    ${tableShell('perCrop', cols)}</div>`;
}

function perClassBlock(rows) {
  if (!rows.length) return '';
  const crops = [...new Set(rows.map(r => r.crop))].sort();
  const kinds = KIND_ORDER.filter(k => rows.some(r => r.kind === k));
  const cols = [
    ['label', 'Class', '', r => txt(r.label)],
    ['kind', 'Kind', '', r => kindBadge(r.kind)],
    ['precision', 'Precision', 'num', r => pct(r.precision)],
    ['recall', 'Recall', 'num', r => pct(r.recall)],
    ['f1', 'F1-score', 'num', r => f1Cell(r.f1)],
    ['support', 'Support', 'num', r => num(r.support)],
  ];
  state.tables.perClass = {
    rows, cols, sort: { key: null, dir: 1 },
    filter: () => {
      const q = $('#pcSearch').value.trim().toLowerCase(), crop = $('#pcCrop').value, kind = $('#pcKind').value;
      return r => (!q || String(r.label).toLowerCase().includes(q)) && (!crop || r.crop === crop) && (!kind || r.kind === kind);
    },
    onRender: n => ($('#pcCount').textContent = `Showing ${n} of ${rows.length} classes`),
  };
  return `<div class="card"><h3 class="card-title">Per-class results <span class="muted small">${rows.length} classes · click a column to sort</span></h3>
    <div class="table-tools">
      <label class="search search-sm"><span class="visually-hidden">Search classes</span>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10.5 3a7.5 7.5 0 0 1 6 12l4.3 4.3-1.5 1.5-4.3-4.3A7.5 7.5 0 1 1 10.5 3zm0 2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11z" fill="currentColor"/></svg>
        <input type="search" id="pcSearch" placeholder="Search classes" autocomplete="off"></label>
      <label class="select"><span>Crop</span><select id="pcCrop"><option value="">All crops</option>${crops.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('')}</select></label>
      <label class="select"><span>Kind</span><select id="pcKind"><option value="">All kinds</option>${kinds.map(k => `<option value="${k}">${kindInfo(k)[0]}</option>`).join('')}</select></label>
      <span id="pcCount" class="muted small" aria-live="polite"></span>
    </div>
    ${tableShell('perClass', cols)}</div>`;
}

// ---------- lightbox ----------
function openLightbox(src, caption) {
  const dlg = $('#lightbox');
  if (typeof dlg.showModal !== 'function') { window.open(src, '_blank', 'noopener'); return; }
  $('#lbImg').src = src;
  $('#lbImg').alt = caption;
  $('#lbCap').textContent = caption;
  $('#lbOpen').href = src;
  dlg.showModal();
}

// ---------- wiring ----------
function init() {
  const input = $('#fileInput'), dz = $('#dropzone');
  input.addEventListener('change', () => pickFile(input.files[0]));

  dz.addEventListener('click', e => { if (!e.target.closest('button') && !state.busy) input.click(); });
  ['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(t => dz.addEventListener(t, () => dz.classList.remove('drag')));
  dz.addEventListener('drop', e => { if (!state.busy) pickFile(e.dataTransfer.files[0]); });
  // stop the browser from navigating away when a file is dropped outside the zone
  ['dragover', 'drop'].forEach(t => window.addEventListener(t, e => e.preventDefault()));
  document.addEventListener('paste', e => {
    const f = [...((e.clipboardData && e.clipboardData.files) || [])][0];
    if (f && !state.busy) pickFile(f);
  });

  document.addEventListener('click', e => {
    const t = e.target.closest('[data-action],[data-sample],[data-sample-kind],[data-lightbox],[role="tab"],[data-sort]');
    if (!t) return;
    if (t.dataset.sample) return useSample(t.dataset.sample, t.dataset.label);
    if (t.dataset.sampleKind) { state.sampleKind = t.dataset.sampleKind; renderSampleFilters(); return renderSamples(); }
    if (t.dataset.lightbox) return openLightbox(t.dataset.lightbox, t.dataset.caption || '');
    if (t.getAttribute('role') === 'tab') return selectTab(t);
    if (t.dataset.sort) return sortTable(t.dataset.table, t.dataset.sort);
    const actions = {
      choose: () => !state.busy && input.click(),
      analyze: () => analyze(false),
      force: () => analyze(true),
      reset: resetAll,
      print: () => window.print(),
      metrics: loadMetrics,
      crops: loadCrops,
      'close-lightbox': () => $('#lightbox').close(),
    };
    (actions[t.dataset.action] || (() => {}))();
  });

  // live filters: crop search and the per-class table controls
  document.addEventListener('input', e => {
    if (e.target.id === 'cropSearch') renderCrops();
    else if (['pcSearch', 'pcCrop', 'pcKind'].includes(e.target.id)) renderTable('perClass');
  });

  // arrow keys move between tabs (WAI-ARIA tabs pattern)
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && $('#lightbox').open) $('#lightbox').close();
    const tab = e.target.closest && e.target.closest('[role="tab"]');
    if (!tab || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
    const tabs = $$('[role="tab"]', tab.parentElement);
    const k = tabs.indexOf(tab) + (e.key === 'ArrowRight' ? 1 : -1);
    selectTab(tabs[(k + tabs.length) % tabs.length]);
  });

  const lb = $('#lightbox');
  lb.addEventListener('click', e => { if (e.target === lb) lb.close(); }); // click on backdrop
  $('#toast').addEventListener('click', e => (e.currentTarget.hidden = true));
  // hide figures whose image file does not exist yet (e.g. training still running)
  document.addEventListener('error', e => {
    const fig = e.target.closest && e.target.closest('.fig');
    if (fig) fig.hidden = true;
  }, true);

  checkHealth();
  setInterval(checkHealth, 30000);
  loadSamples();
  loadCrops();
  loadMetrics();
  // re-fetch when the Model Performance section comes into view and the data is > 30 s old (newly finished models appear)
  if ('IntersectionObserver' in window) new IntersectionObserver(es => {
    if (es.some(e => e.isIntersecting) && Date.now() - (state.metricsAt || 0) > 30000) loadMetrics(true);
  }).observe($('#performance'));
}

init();
