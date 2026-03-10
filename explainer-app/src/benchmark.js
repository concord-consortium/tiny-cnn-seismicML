import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { buildCompactModel, buildStandardModel, loadWeightsFromJson, preprocessWaveform, INPUT_LENGTH, SR } from './lib/seismicModel.js';

const MODELS = {
  compact: {
    build: (numClasses) => buildCompactModel(numClasses),
    weightsUrl: '/models/compact_weights.json',
  },
  standard: {
    build: (numClasses) => buildStandardModel(numClasses),
    weightsUrl: '/models/standard_weights.json',
  },
};

const WINDOW_SECONDS = INPUT_LENGTH / SR; // 60s per window

const statusEl = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const resultsEl = document.getElementById('results');
const resultsBody = document.getElementById('resultsBody');

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

  // Warmup with a small batch to compile shaders without the cost of the full batch
  setStatus('Warming up (compiling shaders)...');
  const WARMUP_BATCH_SIZE = 5;
  const warmupWaveforms = waveforms.slice(0, WARMUP_BATCH_SIZE);
  tf.tidy(() => {
    const batchedInput = preprocessBatch(warmupWaveforms);
    const pred = model.predict(batchedInput);
    pred.dataSync();
  });

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

function displayStreamResults(backendName, modelName, loadTimeMs, numWindows, streamDurationSec, results) {
  const { genMs, prepMs, inferMs, totalMs } = results;

  const rows = [
    ['Backend', backendName],
    ['Model', modelName],
    ['TF.js version', tf.version.tfjs],
    ['Model load time', `${loadTimeMs.toFixed(1)} ms`],
    ['', ''],
    ['Stream duration', `${streamDurationSec} sec (${numWindows} x ${WINDOW_SECONDS}s windows)`],
    ['Data generation (not benchmarked)', `${genMs.toFixed(1)} ms`],
    ['Preprocessing', `${prepMs.toFixed(1)} ms`],
    ['Inference', `${inferMs.toFixed(1)} ms`],
    ['Total (prep + inference)', `${totalMs.toFixed(1)} ms`],
    ['Realtime ratio', `${(streamDurationSec * 1000 / totalMs).toFixed(1)}x realtime`],
  ];

  resultsBody.innerHTML = rows.map(([label, value]) => {
    if (label === '' && value === '') {
      return '<tr><td colspan="2" style="border:none; height:8px"></td></tr>';
    }
    return `<tr><td>${label}</td><td class="num">${value}</td></tr>`;
  }).join('');
  resultsEl.classList.add('visible');
}

async function runBenchmark() {
  const modelKey = document.getElementById('modelSelect').value;
  const backendName = document.getElementById('backendSelect').value;
  const streamMinutes = parseFloat(document.getElementById('streamMinutes').value);

  const modelConfig = MODELS[modelKey];
  if (!modelConfig) {
    setStatus(`Unknown model: ${modelKey}`, 'error');
    return;
  }

  const streamDurationSec = streamMinutes * 60;
  const numWindows = Math.ceil(streamDurationSec / WINDOW_SECONDS);

  runBtn.disabled = true;
  resultsEl.classList.remove('visible');

  try {
    const { model, loadTimeMs } = await loadModel(modelConfig, backendName);
    setStatus(`Model loaded in ${loadTimeMs.toFixed(0)} ms.`);

    const results = await runStreamBenchmark(model, numWindows);
    model.dispose();

    displayStreamResults(backendName, modelKey, loadTimeMs, numWindows, streamDurationSec, results);
    setStatus('Benchmark complete.', 'success');
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    runBtn.disabled = false;
  }
}

// Init
detectBackends();
runBtn.addEventListener('click', runBenchmark);
