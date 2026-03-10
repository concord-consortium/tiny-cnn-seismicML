# Seismic CNN Benchmark

Measures how fast TensorFlow.js can process a stream of seismic data in the browser,
across different model architectures and compute backends.

## Design

The benchmark simulates processing a continuous stream of seismic data by dividing it
into 60-second windows (6000 samples at 100Hz) and running inference on all windows
as a single batch.

### What is measured

- **Model load time** — building the TF.js model, loading weights, and running one
  prediction to trigger GPU shader compilation.
- **Preprocessing** — converting raw waveforms to tensors, detrending, normalizing,
  and concatenating into a batch.
- **Inference** — `model.predict()` on the full batch, including `dataSync()` to
  force synchronous GPU completion so timing is accurate.
- **Realtime ratio** — how many seconds of seismic data can be processed per second
  of wall time (e.g., "600x realtime" means 10 minutes of data processed in 1 second).

Data generation time is reported but excluded from the total, since real usage will
read data from sensors or files rather than generating it.

### Batched vs sequential inference

Early testing compared processing windows one at a time vs batching all windows into
a single `model.predict()` call. Batched inference was consistently 3-4x faster on
GPU backends (WebGL/WebGPU) because the GPU parallelizes across the batch dimension.
Even on the CPU backend, batching was slightly faster due to reduced per-call overhead.
Sequential processing was never faster in any configuration, so the benchmark only
uses batched inference.

### Warmup

TF.js compiles GPU shaders lazily the first time it encounters a particular tensor shape.
Without warmup, the first timed run includes this one-time compilation cost, producing
misleadingly slow results. The benchmark runs a single warmup batch of 5 windows before
timing starts — enough to compile shaders without the cost of processing the full batch.

Note: shader compilation is shape-specific. The warmup batch shape `[5, 6000, 1]` may
differ from the timed batch shape (e.g., `[1440, 6000, 1]` for 24 hours), but in
practice the shaders compiled for batch size 5 are reused for larger batches since the
batch dimension doesn't affect the kernel code.

## TF.js backends

- **WebGL** — GPU via WebGL. Default on most browsers, widely supported. On Apple
  Silicon Macs this goes through a Metal translation layer.
- **WebGPU** — GPU via WebGPU API, which maps to Metal natively on Apple Silicon.
  Requires Chrome 113+, Edge, or Firefox (behind flag). Not yet available in Safari.
  Requires the `@tensorflow/tfjs-backend-webgpu` package.
- **CPU** — pure JavaScript, no GPU. Slowest but works everywhere.

## Models

- **Compact** (~9K params) — 3 conv blocks (16, 32, 64 channels), single FC layer.
  This is the architecture used in Derek's explainer app.
- **Standard** (~94K params) — 4 conv blocks (32, 64, 128, 128 channels), two FC
  layers (128 -> 64 -> num_classes). Larger and more accurate.

Both models use dummy random weights for benchmarking. The inference time depends on
the tensor dimensions flowing through the model (number of layers, filter counts,
kernel sizes), not on the weight values, so dummy weights produce identical performance
characteristics to real trained weights.

## Generating dummy weights

The compact model uses Derek's real trained weights, which are checked into the repo.
The standard model uses dummy weights that are **not checked in** (gitignored to avoid
a 1.7MB generated file). You must generate them before benchmarking the standard model:

```bash
# Generate standard weights (required before first benchmark run)
python scripts/generate_dummy_weights.py --model standard

# Or generate both compact and standard
python scripts/generate_dummy_weights.py

# Generate with 3 classes instead of the default 2
python scripts/generate_dummy_weights.py --model standard --num_classes 3
```

## Running the benchmark

```bash
cd explainer-app
npm install
npm run dev
```

Then open http://localhost:5173/benchmark.html (port may vary).

## Architecture notes

The benchmark is a separate Vite entry point (`benchmark.html` + `src/benchmark.js`)
that imports model builders and preprocessing from `src/lib/seismicModel.js`. It does
not use React — just plain JS and DOM manipulation. The Vite config uses multi-page
mode (`build.rollupOptions.input`) to include both the main React app and the benchmark.

### Weight loading

Both models use a custom JSON weight format (not the standard TF.js `model.json` +
binary shards). A Python script (`scripts/export_compact_weights_for_tfjs.py`) exports
PyTorch checkpoint weights with the necessary transposes:

- Conv1d: PyTorch `(outCh, inCh, kernelSize)` -> TF.js `(kernelSize, inCh, outCh)`
- Dense: PyTorch `(outFeatures, inFeatures)` -> TF.js `(inFeatures, outFeatures)`
- BatchNorm: `weight` -> `gamma`, `bias` -> `beta`, `running_mean` -> `moving_mean`,
  `running_var` -> `moving_variance`

The `loadWeightsFromJson()` function is generic — it detects which layers exist in the
JSON by key presence, so the same function works for both compact and standard models.

### Input format

The models expect single-channel input in channelsLast format: `[batch, 6000, 1]`.
This differs from the PyTorch training pipeline which uses 3 channels (E-N-Z seismic
components) in channelsFirst format: `[batch, 3, 6000]`. The browser models use a
single channel for simplicity.
