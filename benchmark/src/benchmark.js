import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { buildCompactModel, buildStandardModel, loadWeightsFromJson, preprocessWaveform, INPUT_LENGTH, SR } from './lib/seismicModel.js';

const MODELS = {
  compact: {
    build: (numClasses) => buildCompactModel(numClasses),
    weightsUrl: './models/compact_weights.json',
  },
  standard: {
    build: (numClasses) => buildStandardModel(numClasses),
    weightsUrl: './models/standard_weights.json',
  },
};

const WINDOW_SECONDS = INPUT_LENGTH / SR; // 60s per window

// Set this to your deployed Apps Script web app URL
const APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbyr1ytD2z-U0oFeTteoa5etX_Kvm8G4ff54C4hdlvtBfNtHMUCFp24Jwdg045kBU6MqKw/exec';

const statusEl = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const autoBtn = document.getElementById('autoBtn');
const resultsEl = document.getElementById('results');
const systemInfoEl = document.getElementById('systemInfo');
const runsHeadEl = document.getElementById('runsHead');
const runsBodyEl = document.getElementById('runsBody');

// Accumulated run results for the multi-column table
const benchmarkRuns = [];

function setStatus(msg, type = 'info') {
  statusEl.textContent = msg;
  statusEl.className = type;
}

// Detect which backends are available
async function detectBackends() {
  const container = document.getElementById('backendSupport');
  const backends = ['webgl', 'webgpu', 'cpu'];
  const results = [];

  for (const name of backends) {
    let available = false;
    if (name === 'cpu') {
      available = true;
    } else if (name === 'webgl') {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      available = gl != null;
    } else if (name === 'webgpu') {
      available = 'gpu' in navigator;
    }
    results.push({ name, available });
  }

  container.innerHTML = 'Backends: ' + results.map(({ name, available }) =>
    `<span class="${available ? 'supported' : 'unsupported'}">${name}: ${available ? 'available' : 'unavailable'}</span>`
  ).join('');

  // Disable unavailable backends in the select
  const select = document.getElementById('backendSelect');
  for (const option of select.options) {
    const r = results.find(b => b.name === option.value);
    if (r && !r.available) {
      option.disabled = true;
      option.textContent += ' (unavailable)';
    }
  }

  return results;
}

function detectOS() {
  const ua = navigator.userAgent;
  if (/CrOS/.test(ua)) return 'ChromeOS';
  if (/iPad|iPhone|iPod/.test(ua)) return 'iOS';
  if (/Android/.test(ua)) return 'Android';
  if (/Mac OS X/.test(ua)) return 'Mac';
  if (/Windows/.test(ua)) return 'Windows';
  if (/Linux/.test(ua)) return 'Linux';
  return 'Other';
}

function collectSystemMetadata() {
  const meta = {};

  // GPU info via WebGL
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (gl) {
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    if (ext) {
      meta.gpuRenderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL);
      meta.gpuVendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL);
    }
  }

  // CPU cores
  if (navigator.hardwareConcurrency) {
    meta.cpuCores = navigator.hardwareConcurrency;
  }

  // Approximate RAM (Chrome/Edge only)
  if (navigator.deviceMemory) {
    meta.deviceMemoryGB = navigator.deviceMemory;
  }

  // User agent and platform
  meta.userAgent = navigator.userAgent;
  meta.platform = navigator.platform;

  // Screen info
  meta.screenWidth = screen.width;
  meta.screenHeight = screen.height;
  meta.devicePixelRatio = window.devicePixelRatio;

  return meta;
}

function generateDummyWaveform(length = INPUT_LENGTH) {
  const waveform = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    waveform[i] = Math.sin(2 * Math.PI * 5 * i / SR) + (Math.random() - 0.5) * 0.5;
  }
  return waveform;
}

/**
 * Preprocess multiple waveforms into a single batched tensor [N, INPUT_LENGTH, 1].
 *
 * We always use batched inference rather than processing windows one at a time.
 * Testing showed batched predict() is consistently 3-4x faster than sequential
 * on GPU backends (WebGL/WebGPU), because the GPU can parallelize across the
 * batch dimension. Even on the CPU backend, batched is slightly faster due to
 * reduced per-call overhead. There is no scenario where sequential wins.
 */
function preprocessBatch(waveforms) {
  const tensors = waveforms.map(w => preprocessWaveform(w));
  const batched = tf.concat(tensors, 0);
  tensors.forEach(t => t.dispose());
  return batched;
}

/**
 * Build and load the model, return it along with load time.
 */
async function loadModel(modelConfig, backendName) {
  setStatus(`Setting backend to ${backendName}...`);
  await tf.setBackend(backendName);
  await tf.ready();
  setStatus(`Backend ${backendName} ready. Loading model...`);

  const weightsResp = await fetch(modelConfig.weightsUrl);
  if (!weightsResp.ok) throw new Error(`Failed to fetch weights: ${weightsResp.status}`);
  const weightsJson = await weightsResp.json();
  const numClasses = weightsJson.num_classes || 2;

  const loadStart = performance.now();
  const model = modelConfig.build(numClasses);
  loadWeightsFromJson(model, weightsJson);
  // Run one prediction to compile/upload to GPU
  const initInput = preprocessWaveform(generateDummyWaveform());
  const initPred = model.predict(initInput);
  initPred.dataSync();
  initPred.dispose();
  initInput.dispose();
  const loadTimeMs = performance.now() - loadStart;

  return { model, loadTimeMs, numClasses };
}

/**
 * Run stream benchmark: process numWindows windows as a single batch.
 */
async function runStreamBenchmark(model, numWindows) {
  // Generate dummy input data (not part of the benchmark — real data comes from sensors/files)
  setStatus(`Generating ${numWindows} windows of dummy data...`);
  await tf.nextFrame();
  const genStart = performance.now();
  const waveforms = [];
  for (let i = 0; i < numWindows; i++) {
    waveforms.push(generateDummyWaveform());
  }
  const genMs = performance.now() - genStart;

  // Warmup: run several iterations to fully compile shaders and warm GPU caches.
  // A single pass often isn't enough — the second run can still be significantly
  // faster as the GPU settles memory allocation and pipeline caching.
  setStatus('Warming up (compiling shaders)...');
  await tf.nextFrame();
  const WARMUP_BATCH_SIZE = 5;
  const WARMUP_ITERATIONS = 5;
  const warmupWaveforms = waveforms.slice(0, WARMUP_BATCH_SIZE);
  for (let i = 0; i < WARMUP_ITERATIONS; i++) {
    tf.tidy(() => {
      const batchedInput = preprocessBatch(warmupWaveforms);
      const pred = model.predict(batchedInput);
      pred.dataSync();
    });
  }

  // Timed run — measure preprocessing and inference separately
  setStatus(`Processing ${numWindows} windows...`);
  await tf.nextFrame();

  const prepStart = performance.now();
  const batchedInput = preprocessBatch(waveforms);
  const prepMs = performance.now() - prepStart;

  const inferStart = performance.now();
  const pred = model.predict(batchedInput);
  pred.dataSync();
  const inferMs = performance.now() - inferStart;

  batchedInput.dispose();
  pred.dispose();

  return { genMs, prepMs, inferMs, totalMs: prepMs + inferMs };
}

function displaySystemInfo(systemMeta) {
  const rows = [
    ['TF.js version', tf.version.tfjs],
    ['GPU', systemMeta.gpuRenderer || 'unknown'],
    ['CPU cores', systemMeta.cpuCores || 'unknown'],
    ...(systemMeta.deviceMemoryGB ? [['Device memory', `~${systemMeta.deviceMemoryGB} GB`]] : []),
    ['User agent', systemMeta.userAgent],
  ];

  systemInfoEl.innerHTML = rows.map(([label, value]) => {
    const tdClass = String(value).length > 60 ? 'wrap' : '';
    return `<tr><td>${label}</td><td class="${tdClass}">${value}</td></tr>`;
  }).join('');
}

function displayRuns() {
  const runCount = benchmarkRuns.length;

  // Header: Metric | Run 1 | Run 2 | ...
  const headerCells = ['<th>Metric</th>'];
  for (let i = 0; i < runCount; i++) {
    headerCells.push(`<th>Run ${i + 1}</th>`);
  }
  runsHeadEl.innerHTML = `<tr>${headerCells.join('')}</tr>`;

  // Row definitions — each row pulls a value from every run
  const rowDefs = [
    ['Backend', r => r.backendName],
    ['Model', r => r.modelName],
    ['Stream duration', r => `${formatDuration(r.streamDurationSec)} (${r.numWindows} windows)`],
    ['Model load time', r => `${r.loadTimeMs.toFixed(1)} ms`],
    ['Data generation (not timed)', r => `${r.results.genMs.toFixed(1)} ms`],
    ['Preprocessing', r => `${r.results.prepMs.toFixed(1)} ms`],
    ['Inference', r => `${r.results.inferMs.toFixed(1)} ms`],
    ['Total (prep + inference)', r => `${r.results.totalMs.toFixed(1)} ms`],
    ['Realtime ratio', r => `${(r.streamDurationSec * 1000 / r.results.totalMs).toFixed(1)}x`],
  ];

  runsBodyEl.innerHTML = rowDefs.map(([label, fn]) => {
    const cells = [`<td>${label}</td>`];
    for (const run of benchmarkRuns) {
      cells.push(`<td>${fn(run)}</td>`);
    }
    return `<tr>${cells.join('')}</tr>`;
  }).join('');

  resultsEl.classList.add('visible');
}

async function submitToSheet(payload) {
  if (!APPS_SCRIPT_URL) {
    console.warn('APPS_SCRIPT_URL not set — skipping submission');
    return;
  }
  try {
    const resp = await fetch(APPS_SCRIPT_URL, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    const result = await resp.json();
    if (result.status !== 'ok') {
      console.error('Sheet submission error:', result.message);
    }
  } catch (err) {
    console.error('Failed to submit to sheet:', err);
  }
}

/**
 * Run a single benchmark with the given parameters, display results, and submit to sheet.
 * Returns the results object so callers can inspect timing.
 */
async function runSingleBenchmark(modelKey, backendName, streamDurationSec) {
  const modelConfig = MODELS[modelKey];
  if (!modelConfig) {
    throw new Error(`Unknown model: ${modelKey}`);
  }

  const numWindows = Math.ceil(streamDurationSec / WINDOW_SECONDS);

  const { model, loadTimeMs } = await loadModel(modelConfig, backendName);
  setStatus(`Model loaded in ${loadTimeMs.toFixed(0)} ms.`);

  const results = await runStreamBenchmark(model, numWindows);
  model.dispose();

  const systemMeta = collectSystemMetadata();
  displaySystemInfo(systemMeta);
  benchmarkRuns.push({ backendName, modelName: modelKey, loadTimeMs, numWindows, streamDurationSec, results });
  displayRuns();

  const realtimeRatio = streamDurationSec * 1000 / results.totalMs;
  await submitToSheet({
    userName: document.getElementById('userName').value,
    machineLabel: document.getElementById('machineLabel').value,
    os: document.getElementById('osSelect').value,
    backend: backendName,
    model: modelKey,
    tfjsVersion: tf.version.tfjs,
    gpuRenderer: systemMeta.gpuRenderer || '',
    gpuVendor: systemMeta.gpuVendor || '',
    cpuCores: systemMeta.cpuCores || '',
    deviceMemoryGB: systemMeta.deviceMemoryGB || '',
    userAgent: systemMeta.userAgent,
    platform: systemMeta.platform,
    screenWidth: systemMeta.screenWidth,
    screenHeight: systemMeta.screenHeight,
    devicePixelRatio: systemMeta.devicePixelRatio,
    windowSeconds: WINDOW_SECONDS,
    streamDurationSec,
    numWindows,
    modelLoadMs: parseFloat(loadTimeMs.toFixed(1)),
    preprocessingMs: parseFloat(results.prepMs.toFixed(1)),
    inferenceMs: parseFloat(results.inferMs.toFixed(1)),
    totalMs: parseFloat(results.totalMs.toFixed(1)),
    realtimeRatio: parseFloat(realtimeRatio.toFixed(1)),
    dataGenerationMs: parseFloat(results.genMs.toFixed(1)),
  });

  return results;
}

async function runBenchmark() {
  const modelKey = document.getElementById('modelSelect').value;
  const backendName = document.getElementById('backendSelect').value;
  const streamMinutes = parseInt(document.getElementById('streamMinutes').value, 10);
  const streamDurationSec = streamMinutes * 60;

  runBtn.disabled = true;
  autoBtn.disabled = true;
  resultsEl.classList.remove('visible');

  try {
    await runSingleBenchmark(modelKey, backendName, streamDurationSec);
    setStatus('Benchmark complete.', 'success');
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    runBtn.disabled = false;
    autoBtn.disabled = false;
  }
}

async function runAutoBenchmark() {
  runBtn.disabled = true;
  autoBtn.disabled = true;
  resultsEl.classList.remove('visible');

  try {
    // Determine the best available GPU backend, or fall back to CPU
    const backendResults = await detectBackends();
    const hasWebGPU = backendResults.find(b => b.name === 'webgpu')?.available;
    const hasWebGL = backendResults.find(b => b.name === 'webgl')?.available;

    let backend;
    let durations;

    if (hasWebGPU) {
      backend = 'webgpu';
      durations = [24 * 3600]; // start with 24 hours
    } else if (hasWebGL) {
      backend = 'webgl';
      durations = [24 * 3600]; // start with 24 hours
    } else {
      backend = 'cpu';
      durations = [3600]; // 1 hour only
    }

    setStatus(`Auto benchmark: using ${backend} backend...`, 'info');

    // Run first test
    const firstDuration = durations[0];
    setStatus(`Auto benchmark: ${backend} / standard / ${formatDuration(firstDuration)}...`, 'info');
    const results = await runSingleBenchmark('standard', backend, firstDuration);

    // If GPU backend and 24h completed in under 5 seconds, also run 7 days
    if (backend !== 'cpu' && results.totalMs < 5000) {
      const sevenDays = 7 * 24 * 3600;
      setStatus(`Auto benchmark: fast result (${results.totalMs.toFixed(0)} ms), now testing ${backend} / standard / 7 days...`, 'info');
      await runSingleBenchmark('standard', backend, sevenDays);
    }

    setStatus('Auto benchmark complete.', 'success');
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    runBtn.disabled = false;
    autoBtn.disabled = false;
  }
}

function formatDuration(seconds) {
  if (seconds >= 86400) return `${seconds / 86400} days`;
  if (seconds >= 3600) return `${seconds / 3600} hours`;
  return `${seconds / 60} minutes`;
}

// Persist user fields to localStorage
const STORED_FIELDS = ['userName', 'machineLabel', 'osSelect'];
for (const id of STORED_FIELDS) {
  const el = document.getElementById(id);
  const saved = localStorage.getItem(`benchmark_${id}`);
  if (saved !== null) {
    el.value = saved;
  } else if (id === 'osSelect') {
    el.value = detectOS();
  }
  el.addEventListener('change', () => localStorage.setItem(`benchmark_${id}`, el.value));
  if (el.type === 'text') {
    el.addEventListener('input', () => localStorage.setItem(`benchmark_${id}`, el.value));
  }
}

// Init
detectBackends();
runBtn.addEventListener('click', runBenchmark);
autoBtn.addEventListener('click', runAutoBenchmark);
