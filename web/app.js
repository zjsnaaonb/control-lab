/* Control Lab front end: one WebSocket, two canvases, no build step. */

(() => {
  'use strict';

  const MAX_POINTS = 1150;         // 23 s at 50 Hz
  const WINDOW_SEC = 20;           // visible time window
  const COLORS = ['#35c3ab', '#e0a458', '#5aa9e6', '#d977b8'];
  const U_COLOR = '#5aa9e6';

  const SPECS = {
    pid: [
      { key: 'kp', label: 'Kp', min: 0, max: 40, step: 0.1 },
      { key: 'ki', label: 'Ki', min: 0, max: 20, step: 0.1 },
      { key: 'kd', label: 'Kd', min: 0, max: 20, step: 0.1 },
    ],
    lqr: [
      { key: 'q_scale', label: 'Q 权重倍数', min: 0.1, max: 8, step: 0.1 },
      { key: 'r', label: 'R 控制代价', min: 0.05, max: 8, step: 0.05 },
    ],
    mpc: [
      { key: 'horizon', label: '预测步数 N', min: 5, max: 60, step: 1 },
      { key: 'q_scale', label: 'Q 权重倍数', min: 0.1, max: 8, step: 0.1 },
      { key: 'r', label: 'R 控制代价', min: 0.05, max: 8, step: 0.05 },
    ],
  };

  const HINTS = {
    'inverted_pendulum|pid':
      '单回路 PID 只看得见摆角：摆约 2 s 就能扶正，但小车会一直漂走（16 s 已到 20 m）。线性化模型不会因为车跑远而失稳，这既是模型的边界，也是单回路的边界。',
    'inverted_pendulum|lqr':
      'LQR 用全部四个状态反馈，摆角和小车位置一起收敛。但它离线算增益，完全不知道 |u| 有上限。',
    'inverted_pendulum|mpc':
      'MPC 在滚动优化里把 |u| ≤ 限幅当成硬约束，所以扰动下不会出现"饱和后失控"。把 N 调到 5 试试——有限时域会开始抖。',
    'double_integrator|mpc':
      '无阻尼质量块是个纯积分器。把 N 从 30 拖到 8，再看扰动响应：控制量会在限幅边界上来回撞，这就是有限时域预测的典型症状。',
    'double_integrator|pid':
      '二阶对象的 PID 极容易调成欠阻尼。把 Kd 从 1.5 调到 0.3，看振荡怎么变慢——再调回来。',
    'double_integrator|lqr':
      'LQR 解的是无限时域代价，所以哪怕执行机构饱和，恢复过程也比 PID 干净。',
    'mass_spring|pid':
      '欠阻尼振子对相位最敏感。Kd 太小会振荡，太大又会把测量噪声放大——这是微分项的经典两难。',
    'mass_spring|lqr':
      'Riccati 解给出的增益让闭环极点落在稳定区。试着把 R 调大，观察响应变慢、控制量变小。',
    'mass_spring|mpc':
      '振子的自然频率约 1 rad/s。把 N 降到 10 以下，预测窗就短于一个振荡周期，性能会明显掉。',
    'dc_motor|pid':
      '位置伺服最常见的现场配置就是单回路 PID。Kp 决定刚度，Ki 消掉静差，代价是超调。',
    'dc_motor|lqr':
      '电机的角速度通常不可直接测量，这里的 LQR 假设全状态可测；把 R 调大可以看到增益收缩、响应变慢。',
    'dc_motor|mpc':
      '电压限幅是真实电机最常见的约束。把 |u| 降到 1.5，看阶跃响应怎么被"顶住"。',
  };

  const state = {
    plants: [],
    pidDefaults: {},
    plant: 'inverted_pendulum',
    controller: 'mpc',
    params: {
      pid: { kp: 16, ki: 0, kd: 4 },
      lqr: { q_scale: 1, r: 1 },
      mpc: { horizon: 30, q_scale: 1, r: 1 },
    },
    uLimit: 2,
    ref: [],
    history: [],
    paused: false,
    speed: 1,
    connected: false,
    stats: {},
    last: null,
    socket: null,
    sendTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  const dom = {
    conn: $('conn'), clock: $('clock'), solver: $('solver'),
    scene: $('scene'), chart: $('chart'),
    sceneLabel: $('sceneLabel'), chartNote: $('chartNote'),
    hint: $('hint'), plant: $('plant'), plantMeta: $('plantMeta'),
    controller: $('controller'), params: $('params'),
    uLimit: $('uLimit'), uLimitVal: $('uLimitVal'),
    push: $('push'), reset: $('reset'), pause: $('pause'), speed: $('speed'),
    readout: $('readout'),
  };

  const num = (v, d = 2) => {
    if (!Number.isFinite(v)) return '—';
    const s = v.toFixed(d);
    // Avoid "-0.000": a tiny negative residue should read as zero.
    return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
  };

  /* ---------------------------------------------------------------- boot */
  async function boot() {
    try {
      const res = await fetch('/api/plants');
      const data = await res.json();
      state.plants = data.plants;
      state.pidDefaults = data.pid_defaults || {};
      dom.plant.innerHTML = data.plants
        .map((p) => `<option value="${p.key}">${p.name}</option>`)
        .join('');
      dom.plant.value = state.plant;
      resetPidDefaults();
      syncPlantMeta();
      renderParams();
      renderReadout();
      connect();
    } catch (err) {
      dom.conn.textContent = '接口不可用';
      dom.conn.className = 'hud-item offline';
    }
  }

  function currentPlant() {
    return state.plants.find((p) => p.key === state.plant) || null;
  }

  function resetPidDefaults() {
    const d = state.pidDefaults[state.plant] || { kp: 2, ki: 1, kd: 0.2 };
    state.params.pid = { kp: d.kp, ki: d.ki, kd: d.kd };
  }

  function syncPlantMeta() {
    const p = currentPlant();
    if (!p) return;
    dom.sceneLabel.textContent = p.name;
    dom.plantMeta.innerHTML =
      `${p.n_states} 阶状态 · ${p.n_outputs} 个量测<br>` +
      `输入：${p.input_name}${p.unstable ? ' · <span style="color:var(--accent-2)">开环不稳定</span>' : ''}`;
    state.uLimit = p.u_limit;
    dom.uLimit.value = String(p.u_limit);
    dom.uLimitVal.textContent = num(p.u_limit);
    dom.chartNote.textContent = `最近 ${WINDOW_SEC} s`;
    const legend = p.output_names
      .map((n, i) => `<span style="color:${COLORS[i % COLORS.length]}">■</span> ${n.split('(')[0].trim()}`)
      .join('<span style="opacity:.35"> · </span>');
    dom.chartNote.innerHTML = `${legend}<span style="opacity:.35"> · </span><span style="color:${U_COLOR}">■</span> u`;
    dom.hint.textContent = HINTS[`${state.plant}|${state.controller}`] || '调参、施加扰动，看闭环怎么反应。';
  }

  /* ------------------------------------------------------------- sockets */
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${proto}//${location.host}/ws`);
    state.socket = socket;

    socket.onopen = () => {
      state.connected = true;
      dom.conn.textContent = '在线';
      dom.conn.className = 'hud-item online';
      sendConfig(true);
    };

    socket.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'init') {
        applyInit(msg);
      } else if (msg.type === 'samples') {
        appendSamples(msg.data);
        state.stats = msg.stats || {};
        updateHud();
      } else if (msg.type === 'reset_ok') {
        state.history = [];
        state.last = null;
        updateHud();
      }
    };

    socket.onclose = () => {
      state.connected = false;
      dom.conn.textContent = '已断开';
      dom.conn.className = 'hud-item offline';
      setTimeout(connect, 3000);
    };

    socket.onerror = () => socket.close();
  }

  function applyInit(msg) {
    const plant = msg.plant;
    if (plant) {
      state.plant = plant.key;
      dom.plant.value = plant.key;
    }
    if (msg.config && Array.isArray(msg.config.ref)) state.ref = msg.config.ref;
    if (msg.config) {
      state.uLimit = msg.config.u_limit || state.uLimit;
      dom.uLimit.value = String(state.uLimit);
      dom.uLimitVal.textContent = num(state.uLimit);
      state.controller = (msg.controller || state.controller).toLowerCase();
      markController();
    }
    syncPlantMeta();
    renderReadout();
  }

  function sendConfig(immediate = false) {
    const fire = () => {
      if (!state.connected) return;
      const payload = {
        type: 'config',
        config: {
          plant: state.plant,
          controller: state.controller,
          params: { ...state.params[state.controller] },
          u_limit: state.uLimit,
        },
      };
      state.socket.send(JSON.stringify(payload));
    };
    clearTimeout(state.sendTimer);
    if (immediate) fire();
    else state.sendTimer = setTimeout(fire, 220);
  }

  /* ------------------------------------------------------------ data path */
  function appendSamples(batch) {
    if (!batch || !batch.length) return;
    for (const s of batch) state.history.push(s);
    if (state.history.length > MAX_POINTS) {
      state.history.splice(0, state.history.length - MAX_POINTS);
    }
    state.last = state.history[state.history.length - 1];
  }

  function updateHud() {
    const st = state.stats || {};
    dom.clock.textContent = `${num(st.t || 0, 2)} s`;
    if (state.controller === 'lqr') {
      dom.solver.textContent = '0.00 ms（增益离线求出）';
    } else if (state.controller === 'pid') {
      dom.solver.textContent = '0.00 ms（无在线优化）';
    } else {
      const last = state.last;
      dom.solver.textContent = st.solve_ms != null
        ? `${num(st.solve_ms, 2)} ms / ${last ? last.iters : 0} 次迭代`
        : '—';
    }
    renderReadout();
  }

  /* --------------------------------------------------------------- panels */
  function markController() {
    [...dom.controller.children].forEach((b) => {
      b.classList.toggle('active', b.dataset.kind === state.controller);
    });
  }

  function renderParams() {
    const spec = SPECS[state.controller] || [];
    const vals = state.params[state.controller];
    dom.params.innerHTML = spec
      .map((s) => {
        const v = vals[s.key];
        return `
        <div class="row">
          <label for="p-${s.key}">${s.label} <b id="v-${s.key}">${num(v, s.step < 1 ? 2 : 0)}</b></label>
          <input type="range" id="p-${s.key}" min="${s.min}" max="${s.max}"
                 step="${s.step}" value="${v}">
        </div>`;
      })
      .join('');

    spec.forEach((s) => {
      const input = $('p-' + s.key);
      const out = $('v-' + s.key);
      input.addEventListener('input', () => {
        const v = Number(input.value);
        vals[s.key] = v;
        out.textContent = num(v, s.step < 1 ? 2 : 0);
        sendConfig();
      });
    });
    markController();
  }

  function renderReadout() {
    const p = currentPlant();
    const last = state.last;
    const st = state.stats || {};
    const rows = [];

    if (p && last) {
      p.state_names.forEach((name, i) => {
        rows.push([name, num(last.x[i], 3)]);
      });
    }
    rows.push(['__sep__', '__sep__']);
    rows.push(['输入 u', num(st.u != null ? st.u : (last ? last.u : NaN), 3)]);
    rows.push(['限幅', `± ${num(state.uLimit, 2)}`]);
    if (state.controller === 'mpc' && last) {
      rows.push(['求解迭代', String(last.iters)]);
    }
    rows.push(['发散次数', String(st.diverged != null ? st.diverged : 0)]);

    dom.readout.innerHTML = rows
      .map(([k, v]) =>
        k === '__sep__'
          ? '<tr class="sep"><td colspan="2"></td></tr>'
          : `<tr><td>${k}</td><td>${v}</td></tr>`)
      .join('');
  }

  /* --------------------------------------------------------------- canvas */
  function fitCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const box = cv.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width));
    const h = Math.max(1, Math.round(box.height));
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  const sceneCache = { half: 2.5 };

  function drawScene() {
    const { ctx, w, h } = fitCanvas(dom.scene);
    ctx.clearRect(0, 0, w, h);

    const p = currentPlant();
    const last = state.last;
    ctx.fillStyle = 'rgba(53,195,171,0.04)';
    ctx.fillRect(0, 0, w, h);

    const baseY = h - 46;
    const cx = w / 2;

    // Track
    ctx.strokeStyle = '#24313d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(30, baseY);
    ctx.lineTo(w - 30, baseY);
    ctx.stroke();

    if (!p) return;

    if (p.key === 'inverted_pendulum') {
      const halfWidth = Math.min(3.2, (w / 2 - 60) / 62);
      const scale = (w / 2 - 60) / Math.max(1.2, halfWidth);
      const cartX = last ? last.x[0] : 0;
      const theta = last ? last.x[2] : 0.15;
      const px = cx + Math.max(-halfWidth, Math.min(halfWidth, cartX)) * scale;

      // rail ticks
      ctx.strokeStyle = '#1b2530';
      ctx.lineWidth = 1;
      for (let m = -Math.floor(halfWidth); m <= Math.floor(halfWidth); m += 1) {
        const x = cx + m * scale;
        ctx.beginPath();
        ctx.moveTo(x, baseY - 4);
        ctx.lineTo(x, baseY + 4);
        ctx.stroke();
      }

      // cart
      const cartW = 58, cartH = 22;
      ctx.fillStyle = '#22303c';
      roundedRect(ctx, px - cartW / 2, baseY - cartH, cartW, cartH, 5);
      ctx.fill();
      ctx.strokeStyle = '#3a4c5c';
      ctx.stroke();

      // wheels
      ctx.fillStyle = '#111920';
      [-18, 18].forEach((dx) => {
        ctx.beginPath();
        ctx.arc(px + dx, baseY - 2, 6, 0, Math.PI * 2);
        ctx.fill();
      });

      // pole
      const L = Math.min(112, h * 0.42);
      const tipX = px + Math.sin(theta) * L;
      const tipY = baseY - cartH - Math.cos(theta) * L;
      ctx.strokeStyle = '#c9d6e2';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(px, baseY - cartH);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();

      ctx.fillStyle = COLORS[1];
      ctx.beginPath();
      ctx.arc(tipX, tipY, 8, 0, Math.PI * 2);
      ctx.fill();

      // pivot
      ctx.fillStyle = '#4b6072';
      ctx.beginPath();
      ctx.arc(px, baseY - cartH, 4, 0, Math.PI * 2);
      ctx.fill();

      // angle arc
      ctx.strokeStyle = 'rgba(224,164,88,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(px, baseY - cartH, 42, -Math.PI / 2, -Math.PI / 2 + theta, theta < 0);
      ctx.stroke();

      drawForceArrow(ctx, px + cartW / 2 + 10, baseY - cartH / 2, last ? last.u : 0, state.uLimit);
      drawAxisLabel(ctx, `x = ${num(cartX, 2)} m`, 12, baseY + 30);
      drawAxisLabel(ctx, `θ = ${num(theta * 180 / Math.PI, 1)}°`, w - 96, baseY + 30);
      return;
    }

    // Generic 1-D slider view
    const pos = last ? last.y[0] : 0;
    const half = Math.max(1.2, Math.abs(pos) * 1.25);
    sceneCache.half = sceneCache.half * 0.85 + half * 0.15;
    const scale = (w / 2 - 40) / sceneCache.half;
    const px = cx + Math.max(-sceneCache.half, Math.min(sceneCache.half, pos)) * scale;

    // zero marker
    ctx.strokeStyle = '#2a3947';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, baseY - 52);
    ctx.lineTo(cx, baseY + 6);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#22303c';
    roundedRect(ctx, px - 26, baseY - 44, 52, 34, 6);
    ctx.fill();
    ctx.strokeStyle = '#3a4c5c';
    ctx.lineWidth = 1;
    ctx.stroke();

    drawForceArrow(ctx, px + 34, baseY - 27, last ? last.u : 0, state.uLimit);
    drawAxisLabel(ctx, `位置 ${num(pos, 2)}`, 12, baseY + 30);
    drawAxisLabel(ctx, `目标 0.00`, w - 96, baseY + 30);
  }

  function roundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawForceArrow(ctx, x, y, u, limit) {
    const maxLen = 46;
    const len = Math.min(maxLen, Math.abs(u) / Math.max(limit, 1e-6) * maxLen);
    if (len < 2) return;
    const dir = Math.sign(u);
    ctx.strokeStyle = dir > 0 ? COLORS[0] : COLORS[2];
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + dir * len, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + dir * len, y);
    ctx.lineTo(x + dir * (len - 7), y - 5);
    ctx.lineTo(x + dir * (len - 7), y + 5);
    ctx.closePath();
    ctx.fill();
  }

  function drawAxisLabel(ctx, text, x, y) {
    ctx.fillStyle = '#6f8296';
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.fillText(text, x, y);
  }

  /* ---------------------------------------------------------------- chart */
  function drawChart() {
    const { ctx, w, h } = fitCanvas(dom.chart);
    ctx.clearRect(0, 0, w, h);
    const p = currentPlant();
    if (!p) return;

    const padL = 58, padR = 16, padT = 8, padB = 18;
    const plotW = w - padL - padR;

    const hist = state.history;
    const tEnd = hist.length ? hist[hist.length - 1].t : 0;
    const tStart = Math.max(0, tEnd - WINDOW_SEC);
    // Grow the window with the run instead of pinning it at full width: at
    // t = 1 s a fixed 20 s axis would squeeze the trace into a sliver.
    const tSpan = Math.max(2, Math.min(WINDOW_SEC, tEnd));
    const mapX = (t) => padL + ((t - tStart) / tSpan) * plotW;

    // One subplot per signal. Metres and radians must not share an axis: a cart
    // drifting 10 m would flatten the angle trace into a straight line and make
    // a failing PID look perfect.
    const panels = [];
    for (let i = 0; i < p.n_outputs; i += 1) panels.push({ kind: 'out', index: i });
    panels.push({ kind: 'u' });

    const gap = 12;
    const weights = panels.map((pn) => (pn.kind === 'u' ? 1.15 : 1));
    const avail = h - padT - padB - gap * (panels.length - 1);
    const unit = avail / weights.reduce((a, b) => a + b, 0);

    let top = padT;
    panels.forEach((pn, pi) => {
      const ph = unit * weights[pi];
      const isLast = pi === panels.length - 1;
      drawFrame(ctx, padL, top, plotW, ph, hist, tStart, tSpan, mapX, isLast);

      if (pn.kind === 'out') {
        drawSignalPanel(ctx, {
          x: padL, y: top, w: plotW, h: ph,
          values: hist.map((s) => ({ t: s.t, v: s.y[pn.index] })),
          color: COLORS[pn.index % COLORS.length],
          title: p.output_names[pn.index],
          dashedZero: true,
          mapX,
        });
      } else {
        drawSignalPanel(ctx, {
          x: padL, y: top, w: plotW, h: ph,
          values: hist.map((s) => ({ t: s.t, v: s.u })),
          color: U_COLOR,
          title: '控制 u',
          fixedRange: [-state.uLimit, state.uLimit],
          limits: state.uLimit,
          mapX,
        });
      }
      top += ph + gap;
    });
  }

  function drawSignalPanel(ctx, opt) {
    const { x, y, w, h } = opt;
    const values = opt.values;
    let lo, hi;

    if (opt.fixedRange) {
      lo = opt.fixedRange[0];
      hi = opt.fixedRange[1];
    } else {
      lo = Infinity; hi = -Infinity;
      for (const q of values) {
        if (q.v < lo) lo = q.v;
        if (q.v > hi) hi = q.v;
      }
      if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
      // Floor the span so a settled loop does not magnify tiny numerical
      // wiggles into what looks like a real oscillation.
      const MIN_SPAN = 0.12;
      if (hi - lo < MIN_SPAN) {
        const mid = (hi + lo) / 2;
        lo = mid - MIN_SPAN / 2;
        hi = mid + MIN_SPAN / 2;
      }
      const vpad = (hi - lo) * 0.16;
      lo -= vpad; hi += vpad;
    }

    const mapY = (v) => y + h - ((v - lo) / (hi - lo)) * (h - 8) - 4;

    if (opt.limits != null) {
      ctx.strokeStyle = 'rgba(224,86,86,0.5)';
      ctx.setLineDash([5, 4]);
      [opt.limits, -opt.limits].forEach((lim) => {
        const py = mapY(lim);
        ctx.beginPath();
        ctx.moveTo(x, py);
        ctx.lineTo(x + w, py);
        ctx.stroke();
      });
      ctx.setLineDash([]);
    }

    if (opt.dashedZero && lo <= 0 && hi >= 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x, mapY(0));
      ctx.lineTo(x + w, mapY(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    strokeSeries(ctx, values, opt.mapX, mapY, opt.color, 1.6);

    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.fillStyle = opt.color;
    ctx.textAlign = 'left';
    ctx.fillText(opt.title.split('(')[0].trim(), x + 6, y + 13);

    const last = values[values.length - 1];
    if (last) {
      ctx.fillStyle = '#8b9aab';
      ctx.textAlign = 'right';
      ctx.fillText(num(last.v, 3), x + w - 6, y + 13);
    }

    ctx.fillStyle = '#5d6c7c';
    ctx.textAlign = 'right';
    ctx.fillText(num(hi, 2), x - 8, y + 12);
    ctx.fillText(num(lo, 2), x - 8, y + h - 4);
    ctx.textAlign = 'left';
  }

  function drawFrame(ctx, x, y, w, h, hist, tStart, tSpan, mapX, showTicks = true) {
    ctx.strokeStyle = '#1b2530';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.beginPath();
    for (let sec = Math.ceil(tStart); sec <= tStart + tSpan; sec += 1) {
      const px = mapX(sec);
      if (px < x || px > x + w) continue;
      ctx.moveTo(px, y);
      ctx.lineTo(px, y + h);
    }
    ctx.stroke();

    // time ticks, only under the bottom subplot
    if (showTicks && hist.length) {
      ctx.fillStyle = '#5d6c7c';
      ctx.font = '10px ui-monospace, Consolas, monospace';
      ctx.textAlign = 'center';
      for (let sec = Math.ceil(tStart); sec <= tStart + tSpan; sec += 2) {
        const px = mapX(sec);
        if (px < x || px > x + w) continue;
        ctx.fillText(`${sec}s`, px, y + h + 12);
      }
      ctx.textAlign = 'left';
    }
  }

  function strokeSeries(ctx, points, mapX, mapY, color, width) {
    if (!points.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    points.forEach((pt, i) => {
      const x = mapX(pt.t);
      const y = mapY(pt.v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Head marker so the current value is readable at a glance.
    const tail = points[points.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(mapX(tail.t), mapY(tail.v), 2.6, 0, Math.PI * 2);
    ctx.fill();
  }

  /* ----------------------------------------------------------------- loop */
  function loop() {
    drawScene();
    drawChart();
    requestAnimationFrame(loop);
  }

  /* --------------------------------------------------------------- events */
  dom.plant.addEventListener('change', () => {
    state.plant = dom.plant.value;
    resetPidDefaults();
    state.history = [];
    state.last = null;
    syncPlantMeta();
    renderParams();
    sendConfig(true);
  });

  dom.controller.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-kind]');
    if (!btn) return;
    state.controller = btn.dataset.kind;
    markController();
    dom.hint.textContent = HINTS[`${state.plant}|${state.controller}`] || '';
    renderParams();
    sendConfig(true);
  });

  dom.uLimit.addEventListener('input', () => {
    state.uLimit = Number(dom.uLimit.value);
    dom.uLimitVal.textContent = num(state.uLimit);
    sendConfig();
  });

  dom.push.addEventListener('click', () => {
    if (!state.connected) return;
    // Scale the kick so it is always meaningful for the current plant: a
    // pendulum needs a force, a position servo needs a velocity change.
    const amp = state.uLimit * 0.9;
    state.socket.send(JSON.stringify({ type: 'push', amp }));
  });

  dom.reset.addEventListener('click', () => {
    state.history = [];
    state.last = null;
    if (state.connected) state.socket.send(JSON.stringify({ type: 'reset' }));
  });

  dom.pause.addEventListener('click', () => {
    state.paused = !state.paused;
    dom.pause.textContent = state.paused ? '继续' : '暂停';
    dom.pause.classList.toggle('active', state.paused);
    if (state.connected) {
      state.socket.send(JSON.stringify({ type: 'pause', paused: state.paused }));
    }
  });

  const SPEEDS = [1, 2, 4, 0.5];
  dom.speed.addEventListener('click', () => {
    const idx = (SPEEDS.indexOf(state.speed) + 1) % SPEEDS.length;
    state.speed = SPEEDS[idx];
    dom.speed.textContent = `${state.speed}×`;
    if (state.connected) {
      state.socket.send(JSON.stringify({ type: 'speed', speed: state.speed }));
    }
  });

  window.addEventListener('resize', () => { /* canvases refit each frame */ });

  boot();
  requestAnimationFrame(loop);
})();
