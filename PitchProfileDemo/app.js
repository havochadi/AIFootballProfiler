'use strict';
/* PitchProfile demo. Replays the stored output of a real analysis (see build_demo_data.py). */
const D = window.DEMO, M = D.match, RUN = D.run;
// Team colours match the kits on screen (red = Manchester United, blue = Chelsea) so the overlay reads at a glance.
const TEAMS = {A: {name: D.teams.A.name, short: D.teams.A.short, colour: '#d6332a'},
               B: {name: D.teams.B.name, short: D.teams.B.short, colour: '#2563eb'}};
const TEAM_IDX = ['A', 'B'];
const app = document.getElementById('app');
// real analysis time / video length on an RTX 3090: measured detection + CPU stages of both halves, plus about 1 min of
// thumbnails and 5 min of action spotting per half (README figures)
const REAL_FACTOR = (RUN.detect_wall_s + 360 * RUN.halves + RUN.cpu_wall_s) / M.duration_s;
const DEMO_SECONDS = 36;           // how long the simulated analysis plays
const MIN_BLOCKED_MIN = 40;        // the analysis is built for a full half (about 45 minutes) or a full match

const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const fmt = (v, d = 0) => v == null || Number.isNaN(v) ? '–' : Number(v).toLocaleString('en-GB', {minimumFractionDigits: d, maximumFractionDigits: d});
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const mins = s => s < 60 ? `${Math.round(s)} s` : s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : `${Math.max(1, Math.round(s / 60))} min`;
const short = p => p.gk ? 'GK' : p.label.startsWith('#') ? p.label.slice(1) : 'U' + p.label.split(' ')[1];
const sum = a => a.reduce((x, y) => x + y, 0);
const lerp = (a, b, t) => a + (b - a) * t;

const S = {
  view: 'upload', file: null, probe: null, manualMin: 45, sample: false,
  tab: 'overview', sel: null, sort: {k: 'visible_s', dir: -1}, team: 'all', q: '', basis: 'total',
  ev: {kind: 'pass', team: 'all', player: 'all', mode: 'routes', sel: -1},   // player defaults to the top passer below
  track: {clip: 0, boxes: true, labels: true, ball: true, sel: null},
  showShort: false,                // also list players seen for under 10 minutes
};
let stop = () => {};               // cancels the running animation of the view being left
let redraw = () => {};             // redraws the canvases of the visible view (on resize)

/* ───────────────────────── pitch drawing ───────────────────────── */
const PW = 105, PH = 68, PAD = 2.5;

function pitch(canvas, underlay) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
  const chh = cw * (PH + 2 * PAD) / (PW + 2 * PAD);
  const W = Math.round(cw * dpr), H = Math.round(chh * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  canvas.style.aspectRatio = `${PW + 2 * PAD} / ${PH + 2 * PAD}`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const s = cw / (PW + 2 * PAD);
  const g = {ctx, s, w: cw, h: chh, X: x => (x + PAD) * s, Y: y => (y + PAD) * s};
  ctx.fillStyle = '#dfeae3'; ctx.fillRect(0, 0, cw, chh);
  for (let i = 0; i < 10; i += 2) { ctx.fillStyle = '#d6e4dc'; ctx.fillRect(g.X(i * 10.5), g.Y(0), 10.5 * s, PH * s); }
  if (underlay) underlay(g);
  pitchLines(g);
  return g;
}

function pitchLines({ctx, s, X, Y}) {
  ctx.strokeStyle = '#7fa391'; ctx.lineWidth = Math.max(1, s * .2); ctx.fillStyle = '#7fa391';
  const rect = (x, y, w, h) => ctx.strokeRect(X(x), Y(y), w * s, h * s);
  rect(0, 0, PW, PH);
  ctx.beginPath(); ctx.moveTo(X(52.5), Y(0)); ctx.lineTo(X(52.5), Y(PH)); ctx.stroke();
  ctx.beginPath(); ctx.arc(X(52.5), Y(34), 9.15 * s, 0, 7); ctx.stroke();
  rect(0, 34 - 20.16, 16.5, 40.32); rect(PW - 16.5, 34 - 20.16, 16.5, 40.32);
  rect(0, 34 - 9.16, 5.5, 18.32); rect(PW - 5.5, 34 - 9.16, 5.5, 18.32);
  rect(-1.8, 34 - 3.66, 1.8, 7.32); rect(PW, 34 - 3.66, 1.8, 7.32);
  for (const [x, y] of [[52.5, 34], [11, 34], [94, 34]]) { ctx.beginPath(); ctx.arc(X(x), Y(y), Math.max(1.5, s * .3), 0, 7); ctx.fill(); }
}

function heatColour(t) {                      // one sequential ramp: pale amber → orange → deep red
  const stops = [[255, 214, 120], [240, 112, 40], [168, 28, 30]];
  const u = clamp(t) * 2, i = Math.min(1, Math.floor(u)), f = u - i;
  return stops[i].map((c, k) => Math.round(lerp(c, stops[i + 1][k], f)));
}
function heatLayer(g, grid) {
  const rows = grid.length, cols = grid[0].length, mx = Math.max(...grid.flat()) || 1;
  const off = document.createElement('canvas'); off.width = cols; off.height = rows;
  const oc = off.getContext('2d'), img = oc.createImageData(cols, rows);
  grid.forEach((row, r) => row.forEach((v, c) => {
    const t = Math.pow(v / mx, .6), [R, G, B] = heatColour(t), i = (r * cols + c) * 4;
    img.data.set([R, G, B, Math.round(255 * clamp(t * 1.5) * .88)], i);
  }));
  oc.putImageData(img, 0, 0);
  const {ctx, X, Y, s} = g;
  ctx.save(); ctx.beginPath(); ctx.rect(X(0), Y(0), PW * s, PH * s); ctx.clip();
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.filter = `blur(${Math.max(2, s * 1.4)}px)`;
  ctx.drawImage(off, X(0), Y(0), PW * s, PH * s);
  ctx.restore();
}
function binGrid(points) {                    // [x,y] list → 20×32 counts
  const grid = Array.from({length: 20}, () => new Array(32).fill(0));
  for (const [x, y] of points) grid[clamp(Math.floor(y / PH * 20), 0, 19)][clamp(Math.floor(x / PW * 32), 0, 31)]++;
  return grid;
}
function arrow(ctx, x1, y1, x2, y2, head) {
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(a - .45), y2 - head * Math.sin(a - .45));
  ctx.lineTo(x2 - head * Math.cos(a + .45), y2 - head * Math.sin(a + .45)); ctx.closePath(); ctx.fill();
}

/* ───────────────────────── tracking clips: boxes on the footage + positions on the pitch ───────────────────────── */
// One clip = 75 s of the system's own output at 12.5 samples/s.
// Each row: [player (-1 = not named), team, x, y, w, h, pitchX, pitchY, track].
function sample(clip, t) {
  const F = clip.frames;
  if (t < F[0].t - .2 || t > F[F.length - 1].t + .2) return null;
  let lo = 0, hi = F.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (F[m].t <= t) lo = m; else hi = m - 1; }
  const f = F[lo], n = F[lo + 1], gap = n ? n.t - f.t : 9, k = n && gap < .3 ? clamp((t - f.t) / gap) : 0;
  const nm = k > 0 ? (n._m || (n._m = new Map(n.r.map(r => [r[8], r])))) : null;
  const rows = f.r.map(r => {
    const q = nm && nm.get(r[8]), o = {r, bx: r[2], by: r[3], bw: r[4], bh: r[5], px: r[6], py: r[7]};
    if (q && Math.abs(q[2] - r[2]) < 150 && Math.abs(q[3] - r[3]) < 150) {     // never smooth across a camera cut
      o.bx = lerp(r[2], q[2], k); o.by = lerp(r[3], q[3], k); o.bw = lerp(r[4], q[4], k); o.bh = lerp(r[5], q[5], k);
      if (r[6] != null && q[6] != null) { o.px = lerp(r[6], q[6], k); o.py = lerp(r[7], q[7], k); }
    }
    return o;
  });
  let ball = f.b;
  if (f.b && n && n.b && k > 0 && Math.abs(n.b[0] - f.b[0]) < 150) {
    const mix = (a, b) => a == null || b == null ? a : lerp(a, b, k);
    ball = [lerp(f.b[0], n.b[0], k), lerp(f.b[1], n.b[1], k), mix(f.b[2], n.b[2]), mix(f.b[3], n.b[3])];
  }
  return {rows, ball};
}

// Top-down view of the clip at time t (pitch coordinates).
function drawMini(canvas, clip, t, sel, opt = {}) {
  const smp = clip && sample(clip, t), g = pitch(canvas), {ctx, X, Y, s} = g;
  const hits = [];
  if (!smp) return {n: 0, hits};
  const r = Math.max(6, s * 1.6);
  let n = 0;
  for (const o of smp.rows) {
    if (o.px == null) continue;
    n++;
    const named = o.r[0] >= 0, cx = X(o.px), cy = Y(o.py);
    ctx.globalAlpha = named ? 1 : .55; ctx.beginPath(); ctx.arc(cx, cy, named ? r : r * .7, 0, 7);
    ctx.fillStyle = opt.teams === false ? '#6b7d76' : TEAMS[TEAM_IDX[o.r[1]]].colour; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#fff'; ctx.stroke(); ctx.globalAlpha = 1;
    if (named && sel === o.r[0]) { ctx.beginPath(); ctx.arc(cx, cy, r + 4, 0, 7); ctx.lineWidth = 3; ctx.strokeStyle = '#facc15'; ctx.stroke(); }
    if (named && opt.labels !== false) { ctx.fillStyle = '#fff'; ctx.font = `600 ${Math.round(r * .9)}px system-ui,sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(short(D.players[o.r[0]]), cx, cy + .5); }
    if (named) hits.push({pi: o.r[0], x: cx, y: cy});
  }
  if (opt.ball !== false && smp.ball && smp.ball[2] != null) {
    ctx.beginPath(); ctx.arc(X(smp.ball[2]), Y(smp.ball[3]), Math.max(3.5, s * .85), 0, 7); ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#182320'; ctx.stroke();
  }
  return {n, hits};
}

// Boxes drawn over the real footage (image coordinates, 1280×720). st.phase gates what the pipeline has "found so far".
function drawOverlay(canvas, t, clip, st) {
  const dpr = window.devicePixelRatio || 1, cw = canvas.clientWidth || 640, ch = cw * 9 / 16;
  const W = Math.round(cw * dpr), H = Math.round(ch * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cw, ch);
  const k = cw / 1280, smp = sample(clip, t), ph = st.phase;
  st.hits = []; st.n = 0; st.named = 0;
  if (!smp) return null;
  ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  ctx.font = `700 ${Math.max(10, Math.round(cw / 70))}px system-ui,sans-serif`;
  const label = (text, x, y, colour) => {
    const w = ctx.measureText(text).width + 8, h = Math.max(15, cw / 55), yy = y - h < 0 ? y + 2 : y - h - 1;
    ctx.fillStyle = colour; ctx.fillRect(x, yy, w, h); ctx.fillStyle = '#fff'; ctx.fillText(text, x + 4, yy + h / 2 + .5);
  };
  if (ph.boxes) for (const o of smp.rows) {
    const pi = o.r[0], named = pi >= 0, col = ph.teams ? TEAMS[TEAM_IDX[o.r[1]]].colour : '#e5e7eb';
    const x = o.bx * k, y = o.by * k, w = o.bw * k, h = o.bh * k;
    st.n++; if (named) st.named++;
    ctx.globalAlpha = named || !ph.teams ? 1 : .6; ctx.lineWidth = named ? 2.2 : 1.2; ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h); ctx.globalAlpha = 1;
    st.hits.push({pi, x, y, w, h});
    if (st.sel === pi && named) { ctx.lineWidth = 3.5; ctx.strokeStyle = '#facc15'; ctx.strokeRect(x - 3, y - 3, w + 6, h + 6); }
    if (ph.labels && named) label(st.sel === pi ? D.players[pi].name : short(D.players[pi]), x, y, col);
  }
  if (ph.ball && smp.ball) {
    ctx.beginPath(); ctx.arc(smp.ball[0] * k, smp.ball[1] * k, Math.max(6, cw / 110), 0, 7);
    ctx.lineWidth = 2.5; ctx.strokeStyle = '#fde047'; ctx.stroke();
  }
  return smp;
}

/* footage: played from the data drive, never copied into this site */
const VID = {url: null};
const usesSampleFootage = () => S.sample || !S.file || /1_720p|manchester/i.test(S.file.name);
function sourceUrl() {
  if (VID.url) return VID.url;
  if (S.file && S.probe && S.probe.url && !S.probe.error && /1_720p|manchester/i.test(S.file.name)) return S.probe.url;
  // opened through serve.py: it streams the footage from the data drive; opened as a file: read it directly
  if (location.protocol.startsWith('http')) return '/footage';
  return encodeURI('file:///' + M.video.path);
}
function footageMissing(box, retry) {
  box.hidden = false;
  box.innerHTML = `<div><b>Can’t open the match footage.</b><br><span class="small">Looked for ${esc(M.video.path.replace(/\//g, '\\'))}. Is the drive connected?</span><br><button class="primary" style="margin-top:10px">Locate ${esc(M.video.name)}</button><input type="file" accept="video/*,.mkv" hidden></div>`;
  const input = box.querySelector('input');
  box.querySelector('button').onclick = () => input.click();
  input.onchange = () => { if (input.files[0]) { VID.url = URL.createObjectURL(input.files[0]); retry(); } };
}
// calls cb for every presented video frame (and after seeks)
function drive(video, cb) {
  let live = true, id = 0;
  const rv = 'requestVideoFrameCallback' in video;
  const tick = () => { if (!live) return; cb(); id = rv ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick); };
  video.addEventListener('seeked', cb); video.addEventListener('loadeddata', cb);
  tick();
  return () => { live = false; video.removeEventListener('seeked', cb); video.removeEventListener('loadeddata', cb); try { rv ? video.cancelVideoFrameCallback(id) : cancelAnimationFrame(id); } catch (e) { /* video already detached */ } };
}
function pickBox(hits, e, canvas) {
  const rc = canvas.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top;
  return hits.filter(h => h.pi >= 0 && x >= h.x - 4 && x <= h.x + h.w + 4 && y >= h.y - 4 && y <= h.y + h.h + 4)
    .sort((a, b) => a.w * a.h - b.w * b.h)[0];
}

/* ───────────────────────── shared helpers ───────────────────────── */
function setStep() {
  const order = ['upload', 'processing', 'results'];
  document.querySelectorAll('.steps li').forEach(li => {
    const i = order.indexOf(li.dataset.step), cur = order.indexOf(S.view);
    li.classList.toggle('active', i === cur); li.classList.toggle('done', i < cur);
  });
}
function go(view) {
  stop(); stop = () => {}; redraw = () => {};
  S.view = view; setStep(); window.scrollTo(0, 0);
  ({upload: viewUpload, processing: viewProcessing, results: viewResults})[view]();
}
function tip() {
  let t = document.querySelector('.tip');
  if (!t) { t = document.createElement('div'); t.className = 'tip'; t.hidden = true; document.body.appendChild(t); }
  return t;
}
function videoSeconds() { return S.sample || !S.file ? M.duration_s : (S.probe && S.probe.duration) || S.manualMin * 60; }
const DAG = '<span class="dag" title="Spotted by the action-spotting video model and credited to the nearest player. These are expected counts and run well below real totals; use them to compare players, not as totals.">†</span>';

/* ───────────────────────── 1 · upload ───────────────────────── */
function viewUpload() {
  const secs = videoSeconds(), have = S.file || S.sample, probe = S.probe || {};
  const unreadable = S.file && probe.error;
  const m = secs / 60, full = D.length[D.length.length - 1];
  const typical = `a half typically gives ${fmt(full.median)} analysed players (range ${full.min}–${full.max} across ${full.halves} analysed halves), ${fmt(full.median600)} of them on screen long enough for a fair comparison`;
  const checks = [];
  if (have) {
    if (m < MIN_BLOCKED_MIN) checks.push(['bad', 'Too short', `${mins(secs)} is shorter than a half. PitchProfile analyses a full half (about 45 minutes) or a full match.`]);
    else checks.push(['ok', 'Length', `${mins(secs)}: ${m > 60 ? 'a full match' : 'a full half'}. ${typical[0].toUpperCase() + typical.slice(1)}.`]);
    if (S.file) {
      if (probe.h) checks.push([probe.h >= 720 ? 'ok' : 'warn', 'Resolution', `${probe.w}×${probe.h}${probe.h >= 720 ? '' : ': below 720p, players will be harder to read'}`]);
      else if (unreadable) checks.push(['info', 'Resolution', 'This browser cannot read the file’s metadata (common for .mkv). Enter the length below.']);
      if (!/\.(mp4|mkv|mov|avi|webm|m4v|ts)$/i.test(S.file.name)) checks.push(['warn', 'Format', 'Unusual file type. The analysis reads mp4, mkv, mov, avi, webm and ts.']);
    }
    checks.push(['info', 'Camera', 'Main broadcast camera with the pitch in view. Close-ups and crowd shots are skipped automatically.']);
  }
  const est = secs * REAL_FACTOR;
  app.innerHTML = `
  <div class="fade">
    <div class="eyebrow">Step 1 of 3</div>
    <h1>Upload a match video</h1>
    <p class="muted" style="max-width:760px">PitchProfile watches broadcast footage, tracks every player it can see, and returns a set of statistics for each of them. The video is the only input: no tracking feeds, event data or line-ups.</p>
    <div class="grid2" style="margin-top:20px">
      <div>
        <div class="card">
          <div class="drop" id="drop">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m10 9.5 5 2.5-5 2.5z"/></svg>
            <div class="big">Drop a football match video here</div>
            <div class="muted small">mp4 · mkv · mov · avi · webm &nbsp;|&nbsp; a full half or a full match</div>
            <p style="margin:16px 0 0"><button class="primary" id="pick">Choose a video</button></p>
            <input type="file" id="file" accept="video/*,.mkv,.ts" hidden>
            <div class="or">or</div>
            <p style="margin:8px 0 0"><button id="sample">Use the sample match: ${esc(M.title)}, ${esc(M.scope)}</button></p>
          </div>
        </div>
        ${have ? `<div class="card fade" style="margin-top:18px">
          <div class="file">
            <span class="eyebrow">Selected</span>
            <span class="name">${S.file ? esc(S.file.name) : `${esc(M.title)} · ${esc(M.scope)} (sample, 720p broadcast)`}</span>
            <span class="eyebrow">Length</span>
            <span>${unreadable ? `<input type="number" id="manual" min="0.5" max="150" step="0.5" value="${S.manualMin}" style="width:90px"> minutes` : `<b class="num">${mmss(secs)}</b> <span class="muted">(${mins(secs)})</span>`}</span>
            ${S.file ? `<span class="eyebrow">Size</span><span>${fmt(S.file.size / 1048576, S.file.size < 1.05e7 ? 1 : 0)} MB</span>` : ''}
          </div>
          <ul class="checks">${checks.map(([k, t, d]) => `<li><span class="ic ${k}">${{ok: '✓', warn: '!', bad: '×', info: 'i'}[k]}</span><span><b>${t}</b> · ${d}</span></li>`).join('')}</ul>
          <div class="est">
            <div><span>Real analysis time (RTX 3090)</span><strong>≈ ${mins(est)}</strong></div>
            <div><span>This demo plays it in</span><strong>≈ ${DEMO_SECONDS} s</strong></div>
          </div>
          <p style="margin:14px 0 0"><button class="primary" id="go" ${m < MIN_BLOCKED_MIN ? 'disabled' : ''}>Analyse match →</button></p>
        </div>` : ''}
      </div>
      <aside class="card">
        <h2>What to upload</h2>
        <p class="muted small">The models are built and tested on whole halves and whole matches, so upload one of these:</p>
        <ul class="small muted" style="padding-left:18px;margin:8px 0 0">
          <li><b style="color:var(--ink)">A full half (about 45 minutes)</b> of broadcast footage. Typically ${fmt(full.median)} players are analysed, and ${fmt(full.median600)} stay on screen long enough for a fair comparison (median over ${full.halves} analysed halves).</li>
          <li><b style="color:var(--ink)">A full match</b> (both halves, about 90 minutes). The two halves are analysed one after the other and merged.</li>
          <li>Broadcast footage is edited, so players leave the picture. Counts describe what the camera showed, not the whole match.</li>
        </ul>
      </aside>
    </div>
  </div>`;
  const input = document.getElementById('file'), drop = document.getElementById('drop');
  document.getElementById('pick').onclick = () => input.click();
  input.onchange = () => input.files[0] && choose(input.files[0]);
  document.getElementById('sample').onclick = () => { releaseUrl(); S.file = null; S.probe = null; S.sample = true; viewUpload(); };
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => e.dataTransfer.files[0] && choose(e.dataTransfer.files[0]));
  const manual = document.getElementById('manual');
  if (manual) manual.onchange = () => { S.manualMin = Math.max(.5, Number(manual.value) || 45); viewUpload(); };
  const goBtn = document.getElementById('go');
  if (goBtn) goBtn.onclick = () => go('processing');
}
function releaseUrl() { if (S.probe && S.probe.url) URL.revokeObjectURL(S.probe.url); }
function choose(file) {
  releaseUrl(); S.file = file; S.sample = false; S.probe = null;
  const v = document.createElement('video'), url = URL.createObjectURL(file);
  let done = false;
  const finish = info => { if (done) return; done = true; S.probe = {url, ...info}; viewUpload(); };
  v.preload = 'metadata'; v.muted = true;
  v.onloadedmetadata = () => finish(isFinite(v.duration) ? {duration: v.duration, w: v.videoWidth, h: v.videoHeight} : {error: true});
  v.onerror = () => finish({error: true});
  setTimeout(() => finish({error: true}), 5000);
  v.src = url;
}

/* ───────────────────────── 2 · analysing (simulated run over real footage and real output) ───────────────────────── */
function viewProcessing() {
  const secs = videoSeconds(), scale = secs / M.duration_s, frames = Math.round(secs * RUN.hz);
  const real = secs * REAL_FACTOR, sampleFootage = usesSampleFootage(), clip = D.clips[1];
  const stages = [
    {t: 'Sampling the video', d: 'The video is read at 12.5 frames per second. Every later step works on these sampled frames.', s: `${RUN.hz} frames per second`, a: 0, b: .08, c: p => `${mmss(p * secs)} / ${mmss(secs)}`},
    {t: 'Detecting and tracking people', d: 'A detector finds every player, goalkeeper and referee in each frame (white boxes). A tracker links the detections into short per-person tracks.', s: 'YOLOv8x detector · BoT-SORT tracker, on the GPU', a: .04, b: .46, c: p => `${fmt(p * frames)} frames`},
    {t: 'Finding camera cuts', d: 'Broadcast footage cuts between cameras. Each cut is found so that no identity is carried across it by mistake.', s: 'Tracking restarts after every cut', a: .04, b: .46, c: p => `${fmt(p * RUN.cuts * scale)} cuts`},
    {t: 'Mapping the pitch', d: 'Pitch lines and corners are located in the image, which maps every player from camera pixels to metres on the pitch (the pitch view on the right).', s: 'Pitch keypoints → image-to-pitch homography', a: .08, b: .52, c: p => `${fmt(p * M.pitch_view * 100)}% of frames`},
    {t: 'Separating the teams', d: 'Shirt colours are clustered into two teams, so the boxes turn red and blue. Referees are told apart by the detector.', s: 'Kit colour clustering; referees by detector class', a: .5, b: .58, c: p => p < 1 ? '…' : '2 teams + referees'},
    {t: 'Reconstructing the ball path', d: 'The ball is small and often hidden. One most-likely path is chosen for each half and short gaps are filled (yellow ring).', s: 'Ball detector + shortest path over each half', a: .54, b: .66, c: p => `${fmt(p * M.ball_seen * 100)}% located`},
    {t: 'Reading shirt numbers, linking players', d: 'Shirt numbers are read from player thumbnails; players without a readable number are matched by appearance across cuts. Boxes gain their number.', s: 'Number reader + appearance matching across cuts', a: .62, b: .82, c: p => `${fmt(p * RUN.numbers_read * scale)} numbers read`},
    {t: 'Detecting events', d: 'Passes, carries, take-ons, interceptions, recoveries and pressures come from positions and the ball path. Shots and tackles come from a video action model.', s: 'Passes, carries, shots, tackles, interceptions, …', a: .76, b: .92, c: p => `${fmt(p * M.events * scale)} events`},
    {t: 'Computing player statistics', d: 'Distance, speed, heatmaps, on-ball and defending numbers are totalled for every identified player and for both teams.', s: 'Physical, on-ball, defending and off-ball measures', a: .9, b: 1, c: p => `${fmt(Math.round(p * M.profiled))} players`},
  ];
  app.innerHTML = `
  <div class="fade">
    <div class="eyebrow">Step 2 of 3</div>
    <h1>Analysing the match…</h1>
    <p class="muted" style="max-width:820px">This is a ${DEMO_SECONDS}-second replay of a real run. A match this long takes about <b>${mins(real)}</b> on an RTX 3090, so the demo plays back the stored result for the sample match over its real footage. Watch the boxes gain team colours, shirt numbers and the ball as each stage completes.</p>
    <div class="proc" style="margin:18px 0">
      <div class="card">
        <h2>${sampleFootage ? 'Match footage' : 'Your video'}</h2>
        <div class="vbox" id="vbox"><video id="pv" muted playsinline preload="auto"></video><canvas id="pov"></canvas><div class="vmsg" id="pmsg" hidden></div></div>
        <dl class="meta">
          <dt>File</dt><dd>${S.file ? esc(S.file.name) : `${esc(M.title)} · ${esc(M.scope)}`}</dd>
          <dt>Length</dt><dd>${mmss(secs)} (${mins(secs)})</dd>
          <dt>Picture</dt><dd>${S.file && S.probe && S.probe.h ? `${S.probe.w}×${S.probe.h}` : '1280×720 · 25 fps broadcast'}</dd>
        </dl>
        <div class="now"><span class="eyebrow">Now running</span><b id="now-t">Starting…</b><p id="now-d" class="muted small" style="margin:4px 0 0"></p></div>
      </div>
      <div class="card">
        <h2>Tracked positions on the pitch</h2>
        <canvas id="live"></canvas>
        <p class="muted small" style="margin:10px 0 0"><span id="live-n">0</span> players mapped from the image onto the pitch. ${sampleFootage ? '' : 'Shown for the sample match.'}</p>
      </div>
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap">
        <b id="pct">0%</b><span class="muted small" id="clock"></span>
        <button id="skip" class="small">Skip to results</button>
      </div>
      <div class="meter" style="margin:8px 0 14px"><i id="bar"></i></div>
      <ol class="stage-list">${stages.map((st, i) => `<li class="stage wait" id="st${i}"><span class="dot"></span><span class="t">${st.t}</span><span class="cnt" id="c${i}"></span><span class="s">${st.s}</span></li>`).join('')}</ol>
      <div id="done" hidden style="margin-top:16px;display:none;align-items:center;gap:14px;flex-wrap:wrap;padding-top:16px;border-top:1px solid var(--line)">
        <span class="ic ok" style="width:28px;height:28px">✓</span><b>Analysis complete</b>
        <button class="primary" id="open" style="margin-left:auto">Open the match report →</button>
      </div>
    </div>
    <p class="muted small" style="margin-top:14px">Real timings for the sample match (${mins(M.duration_s)}, both halves): detection and tracking ${mins(RUN.detect_wall_s)} on the GPU, thumbnails ≈ 1 min, action spotting ≈ 5 min, then ${mins(RUN.cpu_wall_s)} of CPU stages for teams, ball, events, identities and statistics.</p>
  </div>`;

  const v = document.getElementById('pv'), ov = document.getElementById('pov'), msg = document.getElementById('pmsg');
  let videoOk = false;
  const wire = src => {
    v.src = src; videoOk = false;
    v.onloadedmetadata = () => { videoOk = true; msg.hidden = true; if (sampleFootage) v.currentTime = clip.start_s; else v.loop = true; v.play().catch(() => {}); };
    v.onerror = () => { videoOk = false; if (sampleFootage) footageMissing(msg, () => wire(VID.url)); else { msg.hidden = false; msg.textContent = 'This browser cannot preview the file; the analysis does not need it to.'; } };
  };
  if (sampleFootage) wire(sourceUrl());
  else if (S.probe && S.probe.url && !S.probe.error) wire(S.probe.url);
  else { msg.hidden = false; msg.textContent = S.file ? S.file.name : ''; }

  const live = document.getElementById('live');
  let elapsed = 0, last = performance.now(), raf = 0, finished = false;
  const frame = now => {
    elapsed += (now - last) / 1000; last = now;
    const p = clamp(elapsed / DEMO_SECONDS);
    // footage clock: the real video when it plays, otherwise a stand-in clock over the same clip
    let t = clip.start_s + (elapsed % clip.seconds);
    if (sampleFootage && videoOk) {
      if (v.currentTime >= clip.start_s + clip.seconds - .05 || v.currentTime < clip.start_s - .5) v.currentTime = clip.start_s;
      t = v.currentTime;
    }
    const st = {phase: {boxes: p > .05, teams: p > .5, ball: p > .56, labels: p > .72}, sel: null};
    if (sampleFootage && videoOk) drawOverlay(ov, t, clip, st); else ov.getContext('2d').clearRect(0, 0, ov.width, ov.height);
    const mini = p > .2 && sampleFootage ? drawMini(live, clip, t, null, {teams: p > .5, ball: p > .56, labels: p > .72}) : (pitch(live), {n: 0});
    document.getElementById('live-n').textContent = mini.n;

    document.getElementById('bar').style.width = p * 100 + '%';
    document.getElementById('pct').textContent = Math.floor(p * 100) + '%';
    document.getElementById('clock').textContent = `${mmss(p * real)} of ~${mmss(real)} real analysis time`;
    stages.forEach((sg, i) => {
      const sp = clamp((p - sg.a) / (sg.b - sg.a)), el = document.getElementById('st' + i);
      el.className = 'stage ' + (sp >= 1 ? 'done' : sp > 0 ? 'run' : 'wait');
      el.querySelector('.dot').textContent = sp >= 1 ? '✓' : '';
      document.getElementById('c' + i).textContent = sp > 0 ? sg.c(sp) : '';
    });
    const active = [...stages].reverse().find(sg => p >= sg.a && p < sg.b) || stages[stages.length - 1];
    document.getElementById('now-t').textContent = p >= 1 ? 'Done' : active.t;
    document.getElementById('now-d').textContent = p >= 1 ? 'All statistics are ready.' : active.d;
    if (p >= 1 && !finished) {
      finished = true;
      const d = document.getElementById('done'); d.hidden = false; d.style.display = 'flex';
      document.querySelector('h1').textContent = 'Analysis complete';
    }
    raf = requestAnimationFrame(frame);          // keeps the footage overlay running behind the finished list
  };
  raf = requestAnimationFrame(frame);
  stop = () => { cancelAnimationFrame(raf); v.pause(); v.removeAttribute('src'); v.load(); };
  redraw = () => {};
  document.getElementById('skip').onclick = () => go('results');
  document.getElementById('open').onclick = () => go('results');
}

/* ───────────────────────── 3 · match report ───────────────────────── */
const TABS = [['overview', 'Match overview'], ['tracking', 'Live tracking'], ['players', 'Players'], ['events', 'Event map'], ['about', 'About this analysis']];

function viewResults() {
  const secs = videoSeconds(), uploaded = S.file && !S.sample && !usesSampleFootage();
  app.innerHTML = `
  <div class="fade">
    <div class="hero">
      <div><div class="eyebrow">Step 3 of 3 · Match report</div><h1>${esc(M.title)} · ${esc(M.scope)}</h1>
        <div class="muted">${esc(M.competition)} · ${esc(M.date)} · ${mmss(M.duration_s)} of broadcast video analysed · kit groups named from the on-screen scoreboard:
        <b style="color:var(--a)">${esc(TEAMS.A.name)}</b> (red) and <b style="color:var(--b)">${esc(TEAMS.B.name)}</b> (blue)</div></div>
      <div><button id="again">← Analyse another video</button></div>
    </div>
    ${uploaded ? `<div class="note" style="margin-bottom:16px"><b>Demo note:</b> a real analysis of <i>${esc(S.file.name)}</i> (${mins(secs)}) would take about ${mins(secs * REAL_FACTOR)}. The report below is the stored result of the sample match, so you can see exactly what comes out.</div>` : ''}
    <div class="kpis">
      <div class="kpi"><span>Video analysed</span><strong class="num">${mmss(M.duration_s)}</strong></div>
      <div class="kpi"><span>Players analysed (10+ min on screen)</span><strong class="num">${M.profiled}</strong></div>
      <div class="kpi"><span>Events mapped</span><strong class="num">${fmt(M.events)}</strong></div>
      <div class="kpi"><span>Pitch in view</span><strong class="num">${fmt(M.pitch_view * 100)}%</strong></div>
      <div class="kpi"><span>Ball located</span><strong class="num">${fmt(M.ball_seen * 100)}%</strong></div>
    </div>
    <div class="tabs" role="tablist">${TABS.map(([k, l]) => `<button role="tab" data-tab="${k}" aria-selected="${S.tab === k}">${l}</button>`).join('')}</div>
    <div id="tab"></div>
  </div>`;
  document.getElementById('again').onclick = () => { S.file = null; S.probe = null; S.sample = false; go('upload'); };
  document.querySelector('.tabs').onclick = e => { const b = e.target.closest('[data-tab]'); if (b) { setTab(b.dataset.tab); } };
  setTab(S.tab);
}
function setTab(k) {
  stop(); stop = () => {}; S.tab = k;
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === k));
  ({overview: tabOverview, tracking: tabTracking, players: tabPlayers, events: tabEvents, about: tabAbout})[k]();
}
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => redraw(), 120); });

/* ── overview ── */
function tabOverview() {
  const A = D.teams.A, B = D.teams.B;
  const rows = [
    ['Possession (time on ball)', A.possession * 100, B.possession * 100, v => fmt(v) + '%'],
    ['Touches', A.touches, B.touches],
    ['Passes', A.passes, B.passes],
    ['Pass completion', A.completion * 100, B.completion * 100, v => fmt(v) + '%'],
    ['Progressive passes', A.progressive, B.progressive],
    ['Touches in the final third', A.final_third, B.final_third],
    ['Touches in the box', A.box, B.box],
    ['Shots', A.shots, B.shots],
    ['Interceptions', A.interceptions, B.interceptions],
    ['Recoveries', A.recoveries, B.recoveries],
    ['Pressures', A.pressures, B.pressures],
    ['Distance covered, tracked players (km)', A.distance_km, B.distance_km, v => fmt(v, 1)],
  ];
  const cmp = rows.map(([l, a, b, f = v => fmt(v)]) => {
    const t = a + b || 1;
    return `<div class="row"><div class="lab">${l}</div><span class="v" style="color:var(--a)">${f(a)}</span><div class="track"><i style="width:${a / t * 100}%;background:var(--a)"></i><i style="width:${b / t * 100}%;background:var(--b)"></i></div><span class="v" style="color:var(--b)">${f(b)}</span></div>`;
  }).join('');
  const P = D.players;
  const best = (f, filter = () => true) => P.filter(p => !p.short && filter(p)).reduce((m, p) => f(p) > f(m) ? p : m);
  const leaders = [
    ['Most distance', best(p => p.distance_m), p => `${fmt(p.distance_m / 1000, 1)} km`],
    ['Top speed', best(p => p.top_kmh), p => `${fmt(p.top_kmh, 1)} km/h`],
    ['Most passes', best(p => p.on.passes), p => `${fmt(p.on.passes)} passes`],
    ['Best pass completion (≥ 20 passes)', best(p => p.on.pass_completion, p => p.on.passes >= 20), p => `${fmt(p.on.pass_completion * 100)}%`],
    ['Most carries', best(p => p.on.carries), p => `${fmt(p.on.carries)} carries`],
    ['Most pressures', best(p => p.on.pressures), p => `${fmt(p.on.pressures)} pressures`],
    ['Most recoveries', best(p => p.on.recoveries), p => `${fmt(p.on.recoveries)} recoveries`],
    ['Most interceptions', best(p => p.on.interceptions), p => `${fmt(p.on.interceptions)} interceptions`],
  ].map(([l, p, f]) => `<button class="leader" data-p="${P.indexOf(p)}"><span>${l}</span><b>${f(p)}</b><strong><i class="tdot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${TEAMS[p.team].colour};margin-right:6px"></i>${esc(p.name)}</strong></button>`).join('');
  document.getElementById('tab').innerHTML = `
  <div class="cols fade">
    <div class="card">
      <h2>Team comparison</h2>
      <div class="leg"><span><i class="sw" style="background:var(--a)"></i>${esc(TEAMS.A.name)} (attacks right)</span><span><i class="sw" style="background:var(--b)"></i>${esc(TEAMS.B.name)} (attacks left)</span></div>
      <div class="cmp">${cmp}</div>
      <p class="muted small" style="margin:14px 0 0">Counts cover what the broadcast showed. The camera follows the ball, so players and events away from it are under-counted for both teams.</p>
    </div>
    <div style="display:grid;gap:20px;align-content:start">
      <div class="card"><h2>Standout players</h2><p class="muted small">Click a player to open their full statistics.</p><div class="leaders">${leaders}</div></div>
      <div class="card"><h2>Ball activity through the match</h2><p class="muted small">Touches by team in each 5-minute block of video time.</p>${timeline()}</div>
    </div>
  </div>
  <div class="card fade" style="margin-top:20px">
    <h2>Where each team had the ball</h2>
    <p class="muted small">Heatmap of on-ball events (passes, carries, shots, take-ons), each team attacking left to right.</p>
    <div class="maps"><div><b style="color:var(--a)">${esc(TEAMS.A.name)}</b><canvas id="mapA"></canvas></div><div><b style="color:var(--b)">${esc(TEAMS.B.name)}</b><canvas id="mapB"></canvas></div></div>
  </div>`;
  document.querySelector('.leaders').onclick = e => { const b = e.target.closest('[data-p]'); if (b) { S.sel = +b.dataset.p; S.team = 'all'; S.q = ''; S.jump = true; setTab('players'); } };
  redraw = () => {
    for (const [id, ti] of [['mapA', 0], ['mapB', 1]]) {
      const ons = ['pass', 'carry', 'shot', 'take_on'].map(k => D.events.types.indexOf(k));
      const pts = D.events.rows.filter(r => r[2] === ti && ons.includes(r[0])).map(r => [r[4], r[5]]);
      pitch(document.getElementById(id), g => heatLayer(g, binGrid(pts)));
    }
  };
  redraw();
}
function timeline() {
  const W = 560, H = 190, pad = {l: 30, r: 8, t: 8, b: 26}, T = D.timeline, mx = Math.max(...T.map(t => t[0] + t[1]));
  const bw = (W - pad.l - pad.r) / T.length, sy = v => (H - pad.t - pad.b) * v / mx;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Touches by team per five-minute block">`;
  for (const v of [0, Math.round(mx / 2), mx]) { const y = H - pad.b - sy(v); svg += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y}" y2="${y}" stroke="#e1e9e4"/><text x="${pad.l - 6}" y="${y + 4}" font-size="10" fill="#586b65" text-anchor="end">${v}</text>`; }
  const half = pad.l + 9 * bw;     // video time 45:00, where the second half starts
  svg += `<line x1="${half}" x2="${half}" y1="${pad.t}" y2="${H - pad.b}" stroke="#7d8f88" stroke-dasharray="4 4"/><text x="${half + 5}" y="${pad.t + 12}" font-size="10" fill="#586b65">2nd half</text>`;
  T.forEach(([a, b], i) => {
    const x = pad.l + i * bw + bw * .14, w = bw * .72;
    svg += `<rect x="${x}" y="${H - pad.b - sy(a)}" width="${w}" height="${sy(a)}" fill="var(--a)" rx="2"><title>${i * 5}–${i * 5 + 5} min · ${esc(TEAMS.A.name)} ${a} touches</title></rect>`;
    svg += `<rect x="${x}" y="${H - pad.b - sy(a) - sy(b) - 1}" width="${w}" height="${sy(b)}" fill="var(--b)" rx="2"><title>${i * 5}–${i * 5 + 5} min · ${esc(TEAMS.B.name)} ${b} touches</title></rect>`;
    svg += `<text x="${x + w / 2}" y="${H - 8}" font-size="10" fill="#586b65" text-anchor="middle">${i * 5}'</text>`;
  });
  return svg + '</svg>';
}

/* ── live tracking: the real footage with the system's boxes on top ── */
function tabTracking() {
  const T = S.track;
  document.getElementById('tab').innerHTML = `
  <div class="card fade">
    <div style="display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;align-items:center">
      <div><h2>Live tracking on the match footage</h2>
      <p class="muted small" style="margin:2px 0 0">The boxes are the system’s own output, drawn over the original video. Click a player, on the video or on the pitch, to follow them.</p></div>
      <div class="chips" id="clips" role="group" aria-label="Clip">${D.clips.map((c, i) => `<button data-c="${i}" aria-pressed="${T.clip === i}">${mmss(c.start_s)} · ${esc(c.label)}</button>`).join('')}</div>
    </div>
    <div class="track-grid">
      <div>
        <div class="vbox" id="vbox"><video id="tv" muted playsinline preload="auto"></video><canvas id="ov" class="click"></canvas><div class="vmsg" id="vmsg">Loading footage…</div><span class="hud" id="hud"></span></div>
        <div class="rctl">
          <button id="pp" class="primary" aria-label="Play or pause">Pause</button>
          <input type="range" id="scrub" min="0" max="1000" value="0" aria-label="Position in the clip">
          <span class="num" id="time">0:00</span>
          <div class="seg" id="spd" role="group" aria-label="Speed">${[.5, 1, 2].map(v => `<button data-s="${v}" aria-pressed="${v === 1}">${v}×</button>`).join('')}</div>
        </div>
        <div class="chips" id="toggles" style="margin-top:12px" role="group" aria-label="Overlay layers">
          <button data-k="boxes" aria-pressed="${T.boxes}">Player boxes</button><button data-k="labels" aria-pressed="${T.labels}">Shirt numbers</button><button data-k="ball" aria-pressed="${T.ball}">Ball</button>
        </div>
        <p class="muted small" style="margin:10px 0 0"><i class="sw" style="background:var(--a)"></i>${esc(TEAMS.A.name)} <i class="sw" style="background:var(--b);margin-left:8px"></i>${esc(TEAMS.B.name)} · <b>thin translucent box</b> = tracked player whose identity the system could not settle · <b style="color:#a16207">yellow ring</b> = ball · the broadcast cuts away now and then, so boxes drop out with the pitch view.</p>
      </div>
      <div>
        <canvas id="mini"></canvas>
        <p class="muted small" style="margin:8px 0 10px">The same players mapped onto the pitch (${esc(TEAMS.A.name)} attack right).</p>
        <div id="pcard" class="pcard"></div>
      </div>
    </div>
  </div>`;
  const video = document.getElementById('tv'), ov = document.getElementById('ov'), mini = document.getElementById('mini');
  const msg = document.getElementById('vmsg'), scrub = document.getElementById('scrub'), pp = document.getElementById('pp'), hud = document.getElementById('hud');
  const clip = () => D.clips[T.clip];
  const st = {phase: {boxes: T.boxes, teams: true, labels: T.labels, ball: T.ball}, sel: T.sel, hits: []};
  let miniHits = [];

  const card = () => {
    const box = document.getElementById('pcard');
    if (T.sel == null) { box.innerHTML = '<p class="muted small" style="margin:0">Select a player to see their numbers for the match.</p>'; return; }
    const p = D.players[T.sel], on = p.on;
    box.innerHTML = `<div style="display:flex;gap:8px;align-items:center"><i class="tdot" style="display:inline-block;width:11px;height:11px;border-radius:50%;background:${TEAMS[p.team].colour}"></i><b>${esc(p.name)}</b></div>
      <div class="muted small">${p.position} · ${mmss(p.visible_s)} on screen in the match</div>
      <div class="stat-grid" style="margin-top:10px;grid-template-columns:repeat(2,1fr)">
        <div class="stat"><span>Distance</span><strong>${fmt(p.distance_m / 1000, 2)} km</strong></div>
        <div class="stat"><span>Top speed</span><strong>${fmt(p.top_kmh, 1)} km/h</strong></div>
        <div class="stat"><span>Passes</span><strong>${fmt(on.passes)}</strong><small>${on.pass_completion == null ? '–' : fmt(on.pass_completion * 100) + '% completed'}</small></div>
        <div class="stat"><span>Pressures</span><strong>${fmt(on.pressures)}</strong></div></div>
      <p style="margin:10px 0 0"><button id="open-p">Open full statistics →</button></p>`;
    document.getElementById('open-p').onclick = () => { S.sel = T.sel; S.team = 'all'; S.q = ''; S.jump = true; setTab('players'); };
  };
  const select = pi => { T.sel = pi == null || pi < 0 ? null : (T.sel === pi ? null : pi); card(); cb(); };

  const cb = () => {
    const c = clip(), t = video.currentTime;
    if (video.readyState >= 2 && (t >= c.start_s + c.seconds - .05 || t < c.start_s - .5)) { video.currentTime = c.start_s; return; }   // loop the clip
    st.sel = T.sel; Object.assign(st.phase, {boxes: T.boxes, labels: T.labels, ball: T.ball});
    drawOverlay(ov, t, c, st);
    const m = drawMini(mini, c, t, T.sel); miniHits = m.hits;
    scrub.value = clamp((t - c.start_s) / c.seconds) * 1000;
    document.getElementById('time').textContent = `${mmss(Math.max(0, t - c.start_s))} / ${mmss(c.seconds)}`;
    hud.textContent = `${st.n} tracked · ${st.named} named · ${m.n} on pitch`;
  };
  const wire = src => {
    video.src = src; msg.hidden = false; msg.textContent = 'Loading footage…';
    video.onloadedmetadata = () => { video.currentTime = clip().start_s; };
    video.onplaying = () => { msg.hidden = true; pp.textContent = 'Pause'; };
    video.onpause = () => { pp.textContent = 'Play'; };
    video.onerror = () => footageMissing(msg, () => wire(VID.url));
    video.play().catch(() => {});
  };
  wire(sourceUrl());
  const unhook = drive(video, cb);
  stop = () => { unhook(); video.pause(); video.removeAttribute('src'); video.load(); };
  redraw = cb;
  card();

  pp.onclick = () => { if (video.paused) video.play(); else video.pause(); };
  scrub.oninput = () => { const c = clip(); video.currentTime = c.start_s + scrub.value / 1000 * c.seconds; };
  document.getElementById('clips').onclick = e => {
    const b = e.target.closest('[data-c]'); if (!b) return;
    T.clip = +b.dataset.c; document.querySelectorAll('#clips button').forEach(x => x.setAttribute('aria-pressed', x === b));
    video.currentTime = clip().start_s; video.play().catch(() => {});
  };
  document.getElementById('spd').onclick = e => { const b = e.target.closest('[data-s]'); if (!b) return; video.playbackRate = +b.dataset.s; document.querySelectorAll('#spd button').forEach(x => x.setAttribute('aria-pressed', x === b)); };
  document.getElementById('toggles').onclick = e => { const b = e.target.closest('[data-k]'); if (!b) return; T[b.dataset.k] = !T[b.dataset.k]; b.setAttribute('aria-pressed', T[b.dataset.k]); cb(); };
  ov.onclick = e => { const h = pickBox(st.hits, e, ov); select(h ? h.pi : null); };
  mini.onclick = e => {
    const rc = mini.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top;
    const h = miniHits.map(q => ({q, d: Math.hypot(q.x - x, q.y - y)})).sort((a, b) => a.d - b.d)[0];
    select(h && h.d < 24 ? h.q.pi : null);
  };
  ov.onmousemove = e => { const t = tip(), h = pickBox(st.hits, e, ov); ov.style.cursor = h ? 'pointer' : 'default'; if (h) { t.hidden = false; t.textContent = D.players[h.pi].name; t.style.left = e.clientX + 14 + 'px'; t.style.top = e.clientY + 14 + 'px'; } else t.hidden = true; };
  ov.onmouseleave = () => { tip().hidden = true; };
}

/* ── players ── */
const COLS = [
  {k: 'visible_s', l: 'On screen', v: p => p.visible_s, f: p => mmss(p.visible_s)},
  {k: 'dist', l: 'Distance (km)', v: p => p.distance_m / 1000, d: 2},
  {k: 'top', l: 'Top speed (km/h)', v: p => p.top_kmh, d: 1},
  {k: 'sprints', l: 'Sprints', v: p => p.sprints},
  {k: 'touches', l: 'Touches', on: 'touches'},
  {k: 'passes', l: 'Passes', on: 'passes'},
  {k: 'pc', l: 'Pass %', v: p => p.on.pass_completion == null ? null : p.on.pass_completion * 100, rate: true},
  {k: 'prog', l: 'Prog. passes', on: 'progressive_passes'},
  {k: 'key', l: 'Key passes', on: 'key_passes'},
  {k: 'carries', l: 'Carries', on: 'carries'},
  {k: 'takeons', l: 'Take-ons', on: 'take_ons'},
  {k: 'shots', l: 'Shots', on: 'shots'},
  {k: 'int', l: 'Intercept.', on: 'interceptions'},
  {k: 'rec', l: 'Recoveries', on: 'recoveries'},
  {k: 'press', l: 'Pressures', on: 'pressures'},
  {k: 'tackles', l: 'Tackles', on: 'tackles', dag: true, d: 1},
];
function colValue(c, p) {
  if (c.on) return S.basis === 'per90' ? p.per90[c.on] : p.on[c.on];
  return c.v(p);
}
function tabPlayers() {
  const el = document.getElementById('tab');
  el.innerHTML = `
  <div class="fade">
    <div class="tools">
      <div class="seg" id="teamseg" role="group" aria-label="Team">${[['all', 'Both teams'], ['A', TEAMS.A.short], ['B', TEAMS.B.short]].map(([k, l]) => `<button data-t="${k}" aria-pressed="${S.team === k}">${esc(l)}</button>`).join('')}</div>
      <div class="seg" id="basisseg" role="group" aria-label="Basis">${[['total', 'Totals'], ['per90', 'Per 90 min on screen']].map(([k, l]) => `<button data-b="${k}" aria-pressed="${S.basis === k}">${l}</button>`).join('')}</div>
      <input type="search" id="q" placeholder="Search a player or number" value="${esc(S.q)}" aria-label="Search players" style="min-width:210px">
      <label class="check"><input type="checkbox" id="short" ${S.showShort ? 'checked' : ''}> Include short appearances (${D.players.filter(p => p.short).length} players under 10 min)</label>
      <button id="csv" style="margin-left:auto">Download CSV</button>
    </div>
    <div class="twrap"><table class="pt" id="pt"></table></div>
    <p class="muted small" style="margin:10px 0 0">${DAG} Expected counts from the action-spotting video model; they run well below real totals. Per-90 values for players with under 10 minutes on screen (greyed) are unreliable. “Unnumbered” players are tracks whose shirt number could not be read; some may be fragments of a numbered player.</p>
    <div id="detail"></div>
  </div>`;
  const draw = () => {
    const q = S.q.trim().toLowerCase();
    let list = D.players.map((p, i) => ({p, i})).filter(({p}) => (S.showShort || !p.short) && (S.team === 'all' || p.team === S.team) && (!q || p.name.toLowerCase().includes(q) || p.position.toLowerCase().includes(q)));
    const col = COLS.find(c => c.k === S.sort.k);
    list.sort((a, b) => ((colValue(col, a.p) ?? -1) - (colValue(col, b.p) ?? -1)) * S.sort.dir);
    document.getElementById('pt').innerHTML = `<thead><tr><th>Player</th>${COLS.map(c => `<th data-k="${c.k}" class="${S.sort.k === c.k ? 'sorted' : ''}" title="Sort">${c.l}${c.dag ? DAG : ''}${S.sort.k === c.k ? (S.sort.dir < 0 ? ' ↓' : ' ↑') : ''}</th>`).join('')}</tr></thead><tbody>${list.map(({p, i}) => {
      const dim = S.basis === 'per90' && p.visible_s < 600 ? 'style="color:#8a9a94"' : '';
      return `<tr data-i="${i}" class="${S.sel === i ? 'sel' : ''}" ${dim}><td><b><i class="tdot" style="background:${TEAMS[p.team].colour}"></i>${esc(p.name)}</b><small>${p.position}</small></td>${COLS.map(c => {
        const v = colValue(c, p);
        return `<td>${c.f ? c.f(p) : fmt(v, c.d ?? (S.basis === 'per90' && c.on ? 1 : 0))}</td>`;
      }).join('')}</tr>`;
    }).join('') || `<tr><td colspan="${COLS.length + 1}" class="muted" style="text-align:center;padding:30px">No players match.</td></tr>`}</tbody>`;
    detail();
  };
  const detail = () => {
    const box = document.getElementById('detail');
    if (S.sel == null) { box.innerHTML = '<p class="muted" style="margin-top:18px">Select a player to see their heatmap, speed breakdown and full statistics.</p>'; redraw = () => {}; return; }
    box.innerHTML = playerCard(D.players[S.sel], S.sel);
    const p = D.players[S.sel];
    redraw = () => pitch(document.getElementById('heat'), g => {
      heatLayer(g, p.heat);
      const {ctx, X, Y, s} = g; ctx.beginPath(); ctx.arc(X(p.mean[0]), Y(p.mean[1]), Math.max(5, s * 1.2), 0, 7);
      ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = TEAMS[p.team].colour; ctx.stroke();
    });
    redraw();
    document.getElementById('toev').onclick = () => { S.ev = {...S.ev, kind: 'pass', team: p.team, player: String(S.sel), sel: -1}; setTab('events'); };
  };
  document.getElementById('teamseg').onclick = e => { const b = e.target.closest('[data-t]'); if (b) { S.team = b.dataset.t; tabPlayers(); } };
  document.getElementById('basisseg').onclick = e => { const b = e.target.closest('[data-b]'); if (b) { S.basis = b.dataset.b; tabPlayers(); } };
  document.getElementById('q').oninput = e => { S.q = e.target.value; draw(); };
  document.getElementById('csv').onclick = exportCsv;
  document.getElementById('short').onchange = e => { S.showShort = e.target.checked; draw(); };
  document.getElementById('pt').onclick = e => {
    const th = e.target.closest('th[data-k]'), tr = e.target.closest('tr[data-i]');
    if (th) { S.sort = {k: th.dataset.k, dir: S.sort.k === th.dataset.k ? -S.sort.dir : -1}; draw(); }
    else if (tr) { S.sel = +tr.dataset.i; draw(); document.getElementById('detail').scrollIntoView({behavior: 'smooth', block: 'start'}); }
  };
  draw();
  if (S.jump && S.sel != null) { S.jump = false; setTimeout(() => document.getElementById('detail').scrollIntoView({block: 'start'}), 0); }
}
function playerCard(p, i) {
  const per = S.basis === 'per90', on = p.on, pm = p.per90;
  const stat = (l, k, d = 0, dag = false) => {
    const tot = on[k], p90 = pm[k];
    return `<div class="stat"><span>${l}${dag ? DAG : ''}</span><strong class="num">${fmt(per ? p90 : tot, d || (per ? 1 : 0))}</strong><small>${per ? `${fmt(tot, d)} total` : `${fmt(p90, 1)} per 90`}</small></div>`;
  };
  const z = p.zones_s, zt = sum(Object.values(z)) || 1;
  const zc = {walk: '#cfe3d8', jog: '#86c4a6', run: '#34a07a', high_speed: '#e4962f', sprint: '#b3261e'};
  const zl = {walk: 'Walk', jog: 'Jog', run: 'Run', high_speed: 'High speed', sprint: 'Sprint'};
  const hb = (l, v) => `<div class="hbar"><span>${l}</span><div class="tr"><i style="width:${v * 100}%"></i></div><b class="num">${fmt(v * 100)}%</b></div>`;
  return `<div class="card fade detail-card" style="margin-top:20px">
    <div style="display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;align-items:center">
      <div><h2 style="font-size:21px"><i class="tdot" style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${TEAMS[p.team].colour};margin-right:8px"></i>${esc(p.name)}</h2>
      <span class="pill">${p.position} · suggested from tracking</span> ${p.label.startsWith('Unnumbered') ? '<span class="pill warn">shirt number not read</span>' : ''} <span class="pill">${mmss(p.visible_s)} on screen</span></div>
      <button id="toev">Show this player’s events on the map →</button>
    </div>
    <div class="detail">
      <div><canvas id="heat"></canvas><p class="muted small" style="margin:8px 0 0">Where ${esc(p.name)} was seen, attacking left to right. The white marker is the average position (${fmt(p.mean[0])} m, ${fmt(p.mean[1])} m).</p>
        <h3>Where on the pitch</h3>${hb('Defensive third', p.thirds[0])}${hb('Middle third', p.thirds[1])}${hb('Attacking third', p.thirds[2])}${hb('Left lane', p.lanes[0])}${hb('Central lane', p.lanes[1])}${hb('Right lane', p.lanes[2])}${hb('In the box', p.box_share)}</div>
      <div>
        <div class="tools" style="margin-bottom:0"><div class="seg" id="b2" role="group" aria-label="Basis"><button data-b="total" aria-pressed="${!per}">Totals</button><button data-b="per90" aria-pressed="${per}">Per 90 on screen</button></div></div>
        <h3>Physical</h3>
        <div class="stat-grid">
          <div class="stat"><span>Distance</span><strong class="num">${fmt(p.distance_m / 1000, 2)} km</strong><small>${fmt(p.distance_m / p.visible_s * 60, 0)} m per minute on screen</small></div>
          <div class="stat"><span>Top speed</span><strong class="num">${fmt(p.top_kmh, 1)} km/h</strong></div>
          <div class="stat"><span>Sprints</span><strong class="num">${p.sprints}</strong><small>≥ 1 s above 25.2 km/h</small></div>
        </div>
        <div class="zbar" role="img" aria-label="Time in each speed zone">${Object.keys(zc).map(k => `<i style="width:${z[k] / zt * 100}%;background:${zc[k]}" title="${zl[k]}: ${fmt(z[k] / 60, 1)} min"></i>`).join('')}</div>
        <div class="zlab">${Object.keys(zc).map(k => `<span><i class="sw" style="background:${zc[k]}"></i>${zl[k]} ${fmt(z[k] / 60, 1)} min</span>`).join('')}</div>
        <h3>On the ball</h3>
        <div class="stat-grid">${stat('Touches', 'touches')}${stat('Passes', 'passes')}<div class="stat"><span>Pass completion</span><strong class="num">${on.pass_completion == null ? '–' : fmt(on.pass_completion * 100) + '%'}</strong><small>${fmt(on.passes_completed)} completed</small></div>
          ${stat('Progressive passes', 'progressive_passes')}${stat('Key passes', 'key_passes')}${stat('Carries', 'carries')}${stat('Take-ons', 'take_ons')}${stat('Shots', 'shots')}${stat('Receptions', 'receptions')}
          <div class="stat"><span>Time on ball</span><strong class="num">${fmt(on.time_on_ball_s, 0)} s</strong></div></div>
        <h3>Defending</h3>
        <div class="stat-grid">${stat('Interceptions', 'interceptions')}${stat('Recoveries', 'recoveries')}${stat('Pressures', 'pressures')}${stat('Tackles', 'tackles', 1, true)}${stat('Blocks', 'blocks', 1, true)}${stat('Clearances', 'clearances')}</div>
        <h3>Off the ball</h3>
        <div class="stat-grid">${[['High-intensity runs', 'high_intensity_runs'], ['Runs in behind', 'runs_in_behind'], ['Runs into the box', 'box_runs'], ['Overlaps', 'overlaps'], ['Pressing runs', 'pressing_runs'], ['Recovery runs', 'recovery_runs']].map(([l, k]) => `<div class="stat"><span>${l}</span><strong class="num">${p.runs[k] ?? 0}</strong></div>`).join('')}</div>
      </div></div></div>`;
}
document.addEventListener('click', e => {      // basis toggle inside the player card
  const b = e.target.closest('#b2 [data-b]'); if (b) { S.basis = b.dataset.b; tabPlayers(); }
});
function exportCsv() {
  const head = ['team', 'player', 'position', 'seconds_on_screen', 'distance_m', 'top_speed_kmh', 'sprints', 'touches', 'passes', 'passes_completed', 'pass_completion', 'progressive_passes', 'key_passes', 'carries', 'take_ons', 'shots', 'interceptions', 'recoveries', 'pressures', 'tackles_expected', 'blocks_expected', 'clearances', 'high_intensity_runs'];
  const rows = D.players.map(p => [TEAMS[p.team].name, p.name, p.position, p.visible_s, p.distance_m, p.top_kmh, p.sprints, ...['touches', 'passes', 'passes_completed', 'pass_completion', 'progressive_passes', 'key_passes', 'carries', 'take_ons', 'shots', 'interceptions', 'recoveries', 'pressures', 'tackles', 'blocks', 'clearances'].map(k => p.on[k]), p.runs.high_intensity_runs]);
  const csv = [head, ...rows].map(r => r.map(v => v == null ? '' : /[",\n]/.test(v) ? `"${String(v).replace(/"/g, '""')}"` : v).join(',')).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'})); a.download = 'chelsea-v-manchester-united-h1-player-stats.csv'; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ── event map ── */
const EV_LABEL = {pass: 'Passes', carry: 'Carries', shot: 'Shots', tackle: 'Tackles', interception: 'Interceptions', recovery: 'Recoveries', pressure: 'Pressures', take_on: 'Take-ons', clearance: 'Clearances', cross: 'Crosses', header: 'Headers', block: 'Blocks'};
const OUTCOME = ['', 'complete', 'intercepted'];
const OUT_COL = {complete: '#0a7a4f', intercepted: '#8e24aa', none: '#6b7d76'};
function filteredEvents() {
  const {kind, team, player} = S.ev, ti = D.events.types.indexOf(kind);
  return D.events.rows.map((r, i) => ({r, i})).filter(({r}) => r[0] === ti && (team === 'all' || r[2] === (team === 'A' ? 0 : 1)) && (player === 'all' || r[3] === +player));
}
function tabEvents() {
  const E = S.ev;
  const playerOpts = D.players.map((p, i) => ({p, i})).filter(({p, i}) => (!p.short || String(i) === E.player) && (E.team === 'all' || p.team === E.team)).map(({p, i}) => `<option value="${i}" ${String(i) === E.player ? 'selected' : ''}>${esc(p.name)}</option>`).join('');
  document.getElementById('tab').innerHTML = `
  <div class="fade">
    <div class="tools"><div class="chips" id="kinds" role="group" aria-label="Event type">${D.events.types.map(k => `<button data-k="${k}" aria-pressed="${E.kind === k}">${EV_LABEL[k]}${['tackle', 'block', 'header'].includes(k) ? '†' : ''}</button>`).join('')}</div></div>
    <div class="tools">
      <div class="seg" id="teamseg" role="group" aria-label="Team">${[['all', 'Both teams'], ['A', TEAMS.A.short], ['B', TEAMS.B.short]].map(([k, l]) => `<button data-t="${k}" aria-pressed="${E.team === k}">${esc(l)}</button>`).join('')}</div>
      <select id="who" aria-label="Player"><option value="all">All players</option>${playerOpts}</select>
      <div class="seg" id="mode" role="group" aria-label="View"><button data-m="routes" aria-pressed="${E.mode === 'routes'}">Events and routes</button><button data-m="heat" aria-pressed="${E.mode === 'heat'}">Heatmap</button></div>
      <span class="muted small" id="count"></span>
    </div>
    <div class="evmap">
      <div class="card" style="padding:14px"><canvas id="evc"></canvas><div class="legend-o" id="leg"></div><p class="muted small" style="margin:8px 0 0">Every team is shown attacking left to right. Events with no calibrated pitch position are left out. ${M.unattributed_events} of ${M.events} events could not be credited to an analysed player and appear only under “All players”.</p></div>
      <div><div class="evlist" id="list"></div></div>
    </div>
  </div>`;
  const list = filteredEvents();
  document.getElementById('count').textContent = `${list.length} ${EV_LABEL[E.kind].toLowerCase()}`;
  const spot = ['tackle', 'block', 'header'].includes(E.kind);
  document.getElementById('leg').innerHTML = E.kind === 'pass' && E.mode === 'routes'
    ? Object.entries({complete: 'Completed', intercepted: 'Intercepted', none: 'Outcome unknown (origin only)'}).map(([k, l]) => `<span style="--c:${OUT_COL[k]}">${l}</span>`).join('')
    : spot ? `<span style="--c:#a15c00">† Expected counts from the video action spotter; real totals are higher.</span>` : '';
  let pts = [];
  const draw = () => {
    pts = [];
    pitch(document.getElementById('evc'), g => {
      const {ctx, X, Y, s} = g;
      if (E.mode === 'heat') { heatLayer(g, binGrid(list.map(({r}) => [r[4], r[5]]))); return; }
      ctx.lineCap = 'round';
      list.forEach(({r, i}) => {
        const out = OUTCOME[r[8]] || 'none', col = r[0] === 0 ? OUT_COL[out] : TEAMS[r[2] ? 'B' : 'A'].colour;
        const sel = E.sel === i;
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.globalAlpha = sel ? 1 : .72; ctx.lineWidth = sel ? Math.max(3, s * .5) : Math.max(1.4, s * .26);
        if (r[6] != null && r[7] != null) arrow(ctx, X(r[4]), Y(r[5]), X(r[6]), Y(r[7]), sel ? s * 1.6 : s * 1.1);
        ctx.beginPath(); ctx.arc(X(r[4]), Y(r[5]), sel ? Math.max(6, s * 1.2) : Math.max(2.8, s * .55), 0, 7); ctx.fill();
        if (sel) { ctx.strokeStyle = '#facc15'; ctx.lineWidth = 3; ctx.stroke(); }
        ctx.globalAlpha = 1; pts.push({x: X(r[4]), y: Y(r[5]), i});
      });
    });
  };
  redraw = draw; draw();
  const label = r => `${mmss(r[1])} · ${r[3] >= 0 ? D.players[r[3]].name : TEAMS[r[2] ? 'B' : 'A'].name + ' (unattributed)'}${r[0] === 0 ? ' · ' + (OUTCOME[r[8]] || 'outcome unknown') : ''}`;
  document.getElementById('list').innerHTML = list.slice(0, 400).map(({r, i}) => `<button data-i="${i}" class="${E.sel === i ? 'sel' : ''}"><span>${label(r)}</span></button>`).join('') || '<p class="muted" style="padding:16px">No events for this selection.</p>';
  document.getElementById('list').onclick = e => { const b = e.target.closest('[data-i]'); if (b) { E.sel = +b.dataset.i; document.querySelectorAll('#list button').forEach(x => x.classList.toggle('sel', x === b)); draw(); } };
  const c = document.getElementById('evc'), t = tip();
  c.onmousemove = e => {
    const rc = c.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top;
    const h = pts.map(q => ({q, d: Math.hypot(q.x - x, q.y - y)})).sort((a, b) => a.d - b.d)[0];
    if (h && h.d < 14 && E.mode === 'routes') { t.hidden = false; t.textContent = label(D.events.rows[h.q.i]); t.style.left = e.clientX + 14 + 'px'; t.style.top = e.clientY + 14 + 'px'; } else t.hidden = true;
  };
  c.onmouseleave = () => { t.hidden = true; };
  document.getElementById('kinds').onclick = e => { const b = e.target.closest('[data-k]'); if (b) { E.kind = b.dataset.k; E.sel = -1; tabEvents(); } };
  document.getElementById('teamseg').onclick = e => { const b = e.target.closest('[data-t]'); if (b) { E.team = b.dataset.t; E.player = 'all'; E.sel = -1; tabEvents(); } };
  document.getElementById('who').onchange = e => { E.player = e.target.value; E.sel = -1; tabEvents(); };
  document.getElementById('mode').onclick = e => { const b = e.target.closest('[data-m]'); if (b) { E.mode = b.dataset.m; tabEvents(); } };
}

/* ── about ── */
function tabAbout() {
  document.getElementById('tab').innerHTML = `
  <div class="about-grid fade">
    <div class="card"><h2>How the report was made</h2><ol class="pipe">
      <li><div><b>Sample</b><span>12.5 frames per second from the 25 fps video.</span></div></li>
      <li><div><b>Detect and track</b><span>Players, goalkeepers, referees and the ball are detected, then followed within each camera shot (${RUN.cuts} cuts in this match).</span></div></li>
      <li><div><b>Map to the pitch</b><span>Pitch keypoints give an image-to-pitch mapping, so every position is in metres on a 105 × 68 m pitch.</span></div></li>
      <li><div><b>Teams and direction</b><span>Kit colours split two teams; attack direction comes from where each team stands.</span></div></li>
      <li><div><b>Ball and possession</b><span>One ball path over each half; touches and possession spells follow from it.</span></div></li>
      <li><div><b>Events</b><span>Passes, carries, take-ons, interceptions, recoveries, pressures by rules; shots, tackles, blocks, headers by a video action-spotting model.</span></div></li>
      <li><div><b>Identity</b><span>Shirt numbers are read from thumbnails (${fmt(RUN.numbers_read)} readings here), and unnumbered tracks are matched by appearance.</span></div></li>
      <li><div><b>Statistics</b><span>Per player and per team, from positions, ball and events.</span></div></li>
    </ol></div>
    <div class="card"><h2>How accurate is it?</h2>
      <table class="acc"><thead><tr><th>Component</th><th>Measured on held-out ground truth</th></tr></thead><tbody>
        <tr><td>Passes</td><td>Precision ≈ 0.69, recall ≈ 0.70 (±1 s). Per-player pass counts correlate 0.89 with annotations.</td></tr>
        <tr><td>Shots</td><td>Video model: precision 0.68, recall 0.72 on four held-out halves (58 labelled shots, ±2 s).</td></tr>
        <tr><td>Shirt numbers</td><td>80% of 1,211 test tracklets read correctly.</td></tr>
        <tr><td>Naming players</td><td>On three unseen games the right player is named for 52% of visible time, and 95% of the names given are right.</td></tr>
        <tr><td>Attack direction</td><td>8 of 8 halves correct.</td></tr>
        <tr><td>Tackles, blocks, crosses</td><td>Run low: 1 of 26 tackles was found end to end on unseen games.</td></tr>
      </tbody></table>
      <p class="muted small">The pass figures test the event logic on annotated positions, not the whole chain from video. Shot and naming figures are from held-out games.</p>
    </div>
    <div class="card"><h2>Read the numbers with these limits in mind</h2>
      <ul class="small" style="padding-left:18px;margin:8px 0 0;display:grid;gap:7px">
        <li><b>Only what the camera shows.</b> About ${fmt(M.pitch_view * 100)}% of this match was a usable pitch view. Players off screen get no distance or events.</li>
        <li><b>Identity is partial.</b> Roughly half of player time is tied to a named shirt number; the rest sits under “unnumbered” tracks, and some players are split across several of them.</li>
        <li><b>Edge and foreground players can be missed.</b> The Live tracking tab shows it: a player very close to the camera or at the picture edge sometimes gets no box.</li>
        <li><b>The ball has no height,</b> so lofted passes and headers are the least reliable events.</li>
        <li><b>Counts are partial; rates are fairer.</b> Compare players with per-90-on-screen values rather than totals.</li>
        <li><b>Replays</b> shown from a pitch angle may be analysed like live play.</li>
      </ul>
    </div>
    <div class="card"><h2>Run it on your own video</h2>
      <p class="small">The full pipeline runs locally on an NVIDIA GPU: about ${fmt(REAL_FACTOR, 2)}× the video’s length on an RTX 3090 (${mins(M.duration_s * REAL_FACTOR)} for this ${mins(M.duration_s)} match). The detection pass alone took ${mins(RUN.detect_wall_s)}. Everything after it can be re-run in a few minutes from saved evidence.</p>
      <p class="small muted">This demo replays that stored result, so a ten-minute presentation does not wait for a half-hour job.</p>
    </div>
  </div>`;
}

/* ── start ── */
(() => { const top = D.players.reduce((m, p, i) => p.on.passes > D.players[m].on.passes ? i : m, 0); S.ev.team = D.players[top].team; S.ev.player = String(top); })();
document.getElementById('brand').onclick = e => { e.preventDefault(); S.file = null; S.sample = false; go('upload'); };
go('upload');
