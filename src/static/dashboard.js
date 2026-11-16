const el = (id) => document.getElementById(id);
const format = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "—";
const machineName = (id) => id.replace("id_", "ID ");
const conditionName = (label) => ({ normal: "Normal", anomaly: "Anomaly" }[label] ?? "Unknown");
const clipUrl = (clip, action) => `/api/recordings/${encodeURIComponent(clip.filename)}/${action}`;
const player = el("player");
const limitInput = el("limit");
const state = {
  recordings: new Map(),
  machines: new Map(),
  byMachine: new Map(),
  modelAvailable: false,
  current: null,
  waveform: null,
  selection: 0,
  result: null,
  chart: "waveform",
  busy: false,
  replay: null,
  events: [],
};

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ?? "The dashboard request failed.");
  return data;
}

function message(id, text = "", error = false) {
  const element = el(id);
  element.hidden = !text;
  element.textContent = text;
  element.classList.toggle("error", error);
}

function currentLimit() {
  const value = limitInput.valueAsNumber;
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function decision(score, limit) {
  if (score === null) return ["Waiting", "idle"];
  if (limit === null) return ["Score only", "idle"];
  return score >= limit ? ["Alert", "alert"] : ["Within limit", "good"];
}

function badge(element, score, limit) {
  const [text, tone] = decision(score, limit);
  element.className = `status ${tone}`;
  element.textContent = text;
}

function updateControls() {
  const locked = state.busy || state.replay !== null;
  const hasData = state.machines.size > 0;
  el("machine").disabled = !hasData || locked;
  el("condition").disabled = !hasData || locked;
  el("recording").disabled = !state.current || locked;
  limitInput.disabled = !hasData;
  el("inspect").disabled = !state.current || !state.modelAvailable || locked;
  el("replay").disabled = !state.modelAvailable || !state.current ||
    Boolean(state.replay?.stopped) || (state.busy && !state.replay);
  el("inspect").textContent = state.busy ? "Scoring…" : "Run inspection ↗";
  el("replay").textContent = state.replay ? "■ Stop replay" : "▷ Replay shift";
  el("error-view").disabled = !state.result;
}

function refreshResult() {
  const score = state.result?.score ?? null;
  const limit = currentLimit();
  const ceiling = Math.ceil(Math.max(score ?? 0, limit ?? 0, 5) * 1.2);
  el("score").textContent = format(score);
  el("meter-max").textContent = ceiling;
  el("score-fill").style.width = `${100 * (score ?? 0) / ceiling}%`;
  el("limit-marker").hidden = limit === null;
  el("limit-marker").style.left = `${100 * (limit ?? 0) / ceiling}%`;
  badge(el("decision"), score, limit);
  el("known-label").textContent = state.current ? conditionName(state.current.label) : "—";
  drawSignal();
}

function fillRecordings(preferred) {
  const machine = el("machine").value;
  const condition = el("condition").value;
  const clips = (state.byMachine.get(machine) ?? []).filter(
    (clip) => condition === "all" || clip.label === condition
  );
  const options = clips.map((clip) => new Option(
    `${conditionName(clip.label)} · ${clip.filename.split("_").at(-1).split(".")[0]}`,
    clip.filename
  ));
  el("recording").replaceChildren(...options);
  el("clip-count").textContent = `${clips.length.toLocaleString()} recordings`;
  const selected = clips.find((clip) => clip.filename === preferred) ??
    clips.find((clip) => clip.label === "normal") ?? clips[0];
  el("recording").value = selected?.filename ?? "";
  return selected;
}

function setLimitForMachine() {
  const threshold = state.machines.get(el("machine").value)?.demo_limit;
  limitInput.value = Number.isFinite(threshold) ? threshold.toFixed(3) : "";
  el("limit-help").textContent = Number.isFinite(threshold)
    ? "Demo limit from the 90th percentile of this machine’s normal test scores."
    : "Enter a threshold to turn scores into inspection alerts.";
}

function setChart(kind) {
  state.chart = kind;
  for (const key of ["waveform", "error"]) {
    const button = el(`${key}-view`);
    button.classList.toggle("selected", key === kind);
    button.setAttribute("aria-pressed", String(key === kind));
  }
  el("signal-chart").setAttribute("aria-label", kind === "waveform"
    ? "Waveform of the selected recording" : "Reconstruction error over time");
  drawSignal();
}

async function selectRecording(clip, previousResult = null) {
  const selection = ++state.selection;
  state.current = clip ?? null;
  state.result = previousResult;
  state.waveform = null;
  el("filename").textContent = clip?.filename ?? "No recordings match these filters";
  el("recording-meta").textContent = clip ? "Loading waveform…" : "Choose another condition";
  player.pause();
  if (clip) player.src = clipUrl(clip, "audio");
  else player.removeAttribute("src");
  player.load();
  message("inspection-message");
  setChart("waveform");
  refreshResult();
  updateControls();
  if (!clip) return;
  try {
    const waveform = await request(clipUrl(clip, "waveform"));
    if (selection !== state.selection) return;
    state.waveform = waveform;
    el("recording-meta").textContent = `${format(waveform.duration, 1)} s · ${waveform.sample_rate.toLocaleString()} Hz · mono`;
    drawSignal();
  } catch (error) {
    if (selection !== state.selection) return;
    el("recording-meta").textContent = "Waveform unavailable";
    message("inspection-message", error.message, true);
  }
}

function renderHistory() {
  el("event-count").textContent = `${state.events.length} ${state.events.length === 1 ? "event" : "events"}`;
  const rows = state.events.slice().reverse().map((event) => {
    const row = document.createElement("tr");
    for (const text of [event.time, machineName(event.clip.machine), "", format(event.result.score), "", conditionName(event.clip.label)]) {
      row.insertCell().textContent = text;
    }
    row.cells[0].className = row.cells[3].className = "mono";
    const reopen = document.createElement("button");
    reopen.type = "button";
    reopen.className = "reopen";
    reopen.textContent = event.clip.filename;
    reopen.addEventListener("click", () => {
      if (state.busy || state.replay) return;
      el("machine").value = event.clip.machine;
      el("condition").value = "all";
      fillRecordings(event.clip.filename);
      limitInput.value = event.limit ?? "";
      el("limit-help").textContent = "Threshold used for this inspection.";
      selectRecording(event.clip, event.result);
    });
    row.cells[2].append(reopen);
    const status = document.createElement("span");
    badge(status, event.result.score, event.limit);
    row.cells[4].append(status);
    return row;
  });
  el("event-rows").replaceChildren(...rows);
}

async function inspect() {
  if (!state.current || state.busy) return false;
  const clip = state.current;
  state.busy = true;
  updateControls();
  message("inspection-message", state.replay ? `Replay · clip ${state.replay.step} of ${state.replay.length}` : "Running model inference…");
  try {
    state.result = await request(clipUrl(clip, "score"), { method: "POST" });
    state.events.push({
      time: new Date().toLocaleTimeString([], { hourCycle: "h23" }),
      clip,
      result: state.result,
      limit: currentLimit(),
    });
    refreshResult();
    renderHistory();
    if (!state.replay) message("inspection-message");
    return true;
  } catch (error) {
    message("inspection-message", error.message, true);
    return false;
  } finally {
    state.busy = false;
    updateControls();
  }
}

async function replayShift() {
  if (state.replay) {
    state.replay.stopped = true;
    updateControls();
    return;
  }
  const clips = state.byMachine.get(el("machine").value) ?? [];
  const normal = clips.filter((clip) => clip.label === "normal");
  const anomaly = clips.filter((clip) => clip.label === "anomaly");
  // Sample across the recording list in each condition so the replay shows several takes.
  const take = (list, fraction) => list[Math.floor((list.length - 1) * fraction)];
  const sequence = [...new Set([
    take(normal, 0), take(normal, .3), take(anomaly, .2),
    take(normal, .65), take(anomaly, .5), take(anomaly, .85),
  ].filter(Boolean))];
  if (!sequence.length) sequence.push(...clips.slice(0, 6));
  const replay = { stopped: false, step: 0, length: sequence.length };
  state.replay = replay;
  updateControls();
  el("condition").value = "all";
  let completed = true;
  for (const [index, clip] of sequence.entries()) {
    if (replay.stopped) break;
    replay.step = index + 1;
    await selectRecording(fillRecordings(clip.filename));
    if (replay.stopped) break;
    if (!await inspect()) {
      completed = false;
      break;
    }
    if (index < sequence.length - 1 && !replay.stopped) {
      await new Promise((resolve) => setTimeout(resolve, 900));
    }
  }
  state.replay = null;
  updateControls();
  if (completed) message("inspection-message", replay.stopped ? "Replay stopped." : "Shift replay complete.");
}

function canvasContext(canvas) {
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  return { context, width, height };
}

function guides(context, width, height) {
  context.strokeStyle = "#2a2b30";
  context.lineWidth = 1;
  context.beginPath();
  for (let i = 0; i <= 8; i++) {
    const x = i * width / 8;
    context.moveTo(x, 0);
    context.lineTo(x, height);
  }
  for (let i = 1; i < 4; i++) {
    const y = i * height / 4;
    context.moveTo(0, y);
    context.lineTo(width, y);
  }
  context.stroke();
}

function drawSignal() {
  const chart = canvasContext(el("signal-chart"));
  if (!chart) return;
  const { context, width, height } = chart;
  const waveform = state.waveform;
  guides(context, width, height);
  el("signal-detail").textContent = "";
  if (!waveform) {
    el("signal-start").textContent = "0.0 s";
    el("signal-end").textContent = "—";
    return;
  }

  if (state.chart === "error" && state.result) {
    const { errors, first_center_sample: first, hop_samples: hop } = state.result;
    const max = Math.max(...errors, 1);
    context.lineWidth = 1.5;
    context.strokeStyle = "#ffac65";
    context.beginPath();
    for (const [index, value] of errors.entries()) {
      const x = width * index / Math.max(errors.length - 1, 1);
      const y = 5 + (height - 10) * (1 - value / max);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    el("signal-start").textContent = `${format(first / waveform.sample_rate, 2)} s`;
    el("signal-detail").textContent = `Peak window error ${format(Math.max(...errors), 1)}`;
    el("signal-end").textContent = `${format((first + hop * (errors.length - 1)) / waveform.sample_rate, 2)} s`;
    return;
  }

  const mid = height / 2;
  const max = Math.max(.001, ...waveform.peaks.flat().map(Math.abs));
  context.strokeStyle = "#ffac65";
  context.lineWidth = Math.min(2, .7 * width / waveform.peaks.length);
  context.beginPath();
  for (const [index, [low, high]] of waveform.peaks.entries()) {
    const x = width * (index + .5) / waveform.peaks.length;
    const scale = (mid - 4) / max;
    context.moveTo(x, mid - high * scale);
    context.lineTo(x, mid - low * scale);
  }
  context.stroke();
  if (player.currentTime > 0) {
    const x = width * player.currentTime / waveform.duration;
    context.strokeStyle = "#f5f2eb";
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  el("signal-start").textContent = "0.0 s";
  el("signal-end").textContent = `${format(waveform.duration, 1)} s`;
}

function renderMetricBars(machines) {
  const groups = machines.filter((machine) => Number.isFinite(machine.auc)).map((machine) => {
    const group = document.createElement("div");
    group.className = "bar-group mono";
    const label = document.createElement("span");
    label.textContent = machineName(machine.id);
    group.append(label);
    for (const [index, value] of [machine.auc, machine.pauc].entries()) {
      const track = document.createElement("div");
      track.className = index ? "bar-track secondary-bar" : "bar-track";
      const fill = document.createElement("span");
      fill.style.width = `${value * 100}%`;
      track.append(fill);
      const number = document.createElement("span");
      number.className = "bar-value";
      number.textContent = format(value);
      number.setAttribute("aria-label", `${index ? "Partial AUC" : "ROC AUC"} ${format(value)}`);
      group.append(track, number);
    }
    return group;
  });
  el("metric-bars").replaceChildren(...groups);
}

function drawDistribution() {
  const chart = canvasContext(el("distribution"));
  if (!chart) return;
  const { context, width, height } = chart;
  const machine = el("evaluation-machine").value;
  const clips = (state.byMachine.get(machine) ?? []).filter(
    (clip) => clip.label && Number.isFinite(clip.saved_score)
  );
  const limit = state.machines.get(machine)?.demo_limit;
  if (!clips.length) {
    el("distribution-caption").textContent = "This machine has no saved labeled scores.";
    return;
  }
  const values = clips.map((clip) => clip.saved_score);
  if (Number.isFinite(limit)) values.push(limit);
  const min = Math.floor(Math.min(...values));
  const max = Math.max(min + 1, Math.ceil(Math.max(...values)));
  const bins = Array.from({ length: 20 }, () => [0, 0]);
  for (const clip of clips) {
    const index = Math.min(bins.length - 1, Math.floor((clip.saved_score - min) * bins.length / (max - min)));
    bins[index][clip.label === "anomaly" ? 1 : 0]++;
  }

  const peak = Math.max(...bins.flat(), 1);
  const left = 34;
  const top = 17;
  const bottom = height - 29;
  const plotWidth = width - left - 12;
  const slot = plotWidth / bins.length;
  context.font = "11px monospace";
  context.fillStyle = "#aba9a6";
  context.fillText("Clips", 0, 10);
  for (const amount of [0, Math.round(peak / 2), peak]) {
    const y = bottom - (bottom - top) * amount / peak;
    context.strokeStyle = "#35363a";
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(left + plotWidth, y);
    context.stroke();
    context.fillText(amount, 0, y + 4);
  }
  for (const [index, pair] of bins.entries()) {
    for (const [type, amount] of pair.entries()) {
      const barHeight = (bottom - top) * amount / peak;
      context.fillStyle = type ? "#ffac65" : "#83dbc0";
      context.fillRect(left + slot * index + type * slot / 2, bottom - barHeight, Math.max(1, slot / 2 - 1), barHeight);
    }
  }
  if (Number.isFinite(limit)) {
    const x = left + (limit - min) * plotWidth / (max - min);
    context.strokeStyle = "#f5f2eb";
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x, bottom);
    context.stroke();
    context.setLineDash([]);
  }
  context.fillStyle = "#aba9a6";
  context.fillText(min, left, bottom + 15);
  context.fillText(max, left + plotWidth - context.measureText(max).width, bottom + 15);
  const axis = "Anomaly score";
  context.fillText(axis, left + (plotWidth - context.measureText(axis).width) / 2, height - 1);
  el("distribution-caption").textContent = `${clips.length} labeled clips${Number.isFinite(limit) ? ` · Demo limit ${format(limit)}` : ""}`;
}

async function initialize() {
  try {
    const data = await request("/api/catalog");
    state.modelAvailable = data.checkpoint_available;
    state.machines = new Map(data.machines.map((machine) => [machine.id, machine]));
    state.recordings = new Map(data.recordings.map((clip) => [clip.filename, clip]));
    for (const machine of data.machines) state.byMachine.set(machine.id, []);
    for (const clip of data.recordings) state.byMachine.get(clip.machine).push(clip);
    el("checkpoint-name").textContent = state.modelAvailable ? data.checkpoint : "Checkpoint missing";
    el("model-status").classList.add(state.modelAvailable ? "ready" : "unavailable");
    el("recording-count").textContent = data.recordings.length.toLocaleString();
    el("machine-count").textContent = data.machines.length;
    const evaluated = data.machines.filter((machine) => Number.isFinite(machine.auc));
    for (const field of ["auc", "pauc"]) {
      const mean = evaluated.reduce((total, machine) => total + machine[field], 0) / evaluated.length;
      el(`mean-${field}`).textContent = format(mean);
    }
    for (const id of ["machine", "evaluation-machine"]) {
      el(id).replaceChildren(...data.machines.map((machine) => new Option(machineName(machine.id), machine.id)));
    }
    renderMetricBars(data.machines);
    el("download").hidden = !data.scores_available;
    if (!evaluated.length) {
      message("evaluation-message", "Run the evaluate task to generate labeled scores, then restart the dashboard to view them here.");
    }
    if (!data.recordings.length) {
      message("page-error", `No ToyCar recordings found in ${data.recordings_dir}.`, true);
      return;
    }
    if (!state.modelAvailable) message("page-error", `Checkpoint missing: ${data.checkpoint}.`, true);
    setLimitForMachine();
    const initial = fillRecordings();
    await selectRecording(initial);
    if (state.current === initial && !state.events.length && !state.replay && state.modelAvailable) {
      await inspect();
    }
  } catch (error) {
    message("page-error", `Unable to load dashboard data. ${error.message}`, true);
  }
}

el("machine").addEventListener("change", () => {
  setLimitForMachine();
  selectRecording(fillRecordings());
});
el("condition").addEventListener("change", () => selectRecording(fillRecordings()));
el("recording").addEventListener("change", () => selectRecording(state.recordings.get(el("recording").value)));
el("inspect").addEventListener("click", inspect);
el("replay").addEventListener("click", replayShift);
el("waveform-view").addEventListener("click", () => setChart("waveform"));
el("error-view").addEventListener("click", () => setChart("error"));
el("evaluation-machine").addEventListener("change", drawDistribution);
limitInput.addEventListener("input", () => {
  el("limit-help").textContent = "Scores at or above this threshold raise an alert.";
  refreshResult();
});
player.addEventListener("timeupdate", drawSignal);
for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => {
    for (const tab of document.querySelectorAll("[data-view]")) {
      const active = tab === button;
      tab.classList.toggle("selected", active);
      tab.setAttribute("aria-pressed", String(active));
      el(`${tab.dataset.view}-view`).hidden = !active;
    }
    if (button.dataset.view === "evaluation") drawDistribution();
    else drawSignal();
  });
}
const resize = new ResizeObserver(() => {
  drawSignal();
  drawDistribution();
});
resize.observe(el("signal-chart"));
resize.observe(el("distribution"));
initialize();
