# Directly programmed Life inside Qwen

Supplementary study material, source snapshot dated 2026-11-07.

This bundle contains selected source, model metadata, and results from the
Qwen-Life experiment. It accompanies the approachable paper. The reported
experiments were run earlier; source files are a current snapshot and are not
a commit-exact reconstruction of every historical run. The bundle was checked
for Python syntax and archive integrity, but was not independently rerun as a
complete reproduction. It includes no pretrained or merged model weights.

## What is included

- `life/internal_merge.py`, `prototype/internal_circuit.py`, and
  `life/internal_runtime.py`: the current directly programmed internal
  controller, merge, and generic inference loop.
- `life/merge.py` and `prototype/circuit.py`: the earlier three-layer design,
  included so historical evidence can be interpreted in its proper context.
- `reference/`: an independent Life specification and a small standalone
  analytically assigned neural network.
- `validation/`: validators of the real merged checkpoint and semantic routing.
- `ui/`: renderer, bridge, and source-level tests. The browser validator with
  a private default server address is deliberately omitted.
- `metadata/`: checkpoint configuration metadata only, not tensors.
- `reports/`: selected, historically recorded JSON results. Their outcomes,
  prompts, generated text, counts, and hashes are preserved. Machine-specific
  project paths and private demo URLs are replaced by labeled placeholders.
- `MANIFEST.json`: per-file SHA-256 values, original source hashes, redaction
  locations, omissions, and validation scope. The manifest excludes its own
  hash to avoid a recursive definition.

The original checkpoint has 1,215,625,120 parameters and three added layers.
The current internal-controller checkpoint has 4,094,404,032 parameters and
twelve added layers. Do not pool their results into one validation claim.

## Reading the evidence

The standalone reference tests enumerate all 512 binary local neighborhoods.
The intermediate-representation tests check compiled symbolic traces. Neither
is an exhaustive test of the real merged language model. The latter is checked
by the real-checkpoint validators, comparing its predicted tokens and recurrent
board state to an independent oracle.

The v3 core panel passed seven scenarios and 36 isolated private predictions.
The targeted panel passed 13 cases. The broader semantic panel passed 26 of 31
and retains its `failed` status and all five failed cases. The later corner
audit checked 160 generations and 2,746 outputs on three worlds. These results
do not prove all 2^256 boards, all language histories, or infinite execution.

Earlier-model evidence includes 402 generations and a constructed-weight
ablation. This evidence belongs to the earlier three-layer checkpoint, not v3.
The original direct-circuit throughput and current browser Step latency use
different measurement boundaries, so their ratio is not a controlled slowdown.

The loading-state fixture report explicitly did not execute the neural model.
Browser fixtures, live UI tests, and real-model validation remain separate.

## External dependencies and provenance

The source imports two upstream projects, intentionally not vendored here:

| Dependency | Repository | Observed checkout |
| --- | --- | --- |
| qwenputer | https://github.com/divagr18/qwenputer | `bec1a0c9654ac8151b6f36ed0154a19c53141384` |
| qwenGPU | https://github.com/vixhal-baraiya/qwenGPU | `e2695f6a0a818ebc68e2e14b6b990eddf04c51b3` |

Both upstream projects use the MIT license; see their repositories for the
full terms and notices. The observed qwenputer checkout differs from the pin
inside qwenGPU's package metadata, so use the explicit checkouts below.
No upstream source is redistributed in this archive.

The base model is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. Obtain its weights separately from
https://huggingface.co/Qwen/Qwen3.5-0.8B and follow the model's own terms.

The recorded CUDA runs used NVIDIA's PyTorch
`2.9.0a0+145a3a7bda.nv25.10`, CUDA 13.0, and Transformers 5.22.1.
`Dockerfile` records the NVIDIA image tag and dependency requirements. It is
not a fully locked image digest or dependency lockfile. `Dockerfile.fast`
records the optional convolution kernel and the removal of an incompatible
FLA version; it does not claim FLA acceleration was active.

## Starting from this directory

These commands are provided for inspection and follow-up reproduction. They
have not been executed from this exported bundle. GPU steps require compatible
NVIDIA drivers, container support, and sufficient free memory for the widened
checkpoint and temporary construction tensors. Run checks in fresh directories;
do not point validators at a live world.

Obtain the two source dependencies:

```bash
mkdir -p references
git clone https://github.com/divagr18/qwenputer.git references/qwenputer
git -C references/qwenputer checkout bec1a0c9654ac8151b6f36ed0154a19c53141384
git clone https://github.com/vixhal-baraiya/qwenGPU.git references/qwenGPU
git -C references/qwenGPU checkout e2695f6a0a818ebc68e2e14b6b990eddf04c51b3
docker build -t qwen-life:dev .
```

Run the standalone local-rule and analytic intermediate-representation tests:

```bash
docker run --rm -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life/reference qwen-life:dev python -m unittest -v
docker run --rm -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -m pytest -q prototype/test_internal_circuit.py
```

Put the base-model snapshot in `base-qwen/`, then build the internal-controller
checkpoint and run the core real-model validator:

```bash
docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -m life.internal_merge base-qwen build/qwen-life-internal-08b-v3
docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -m validation.internal_e2e build/qwen-life-internal-08b-v3 \
  --runtime-dir runs/new-e2e --out results/new-e2e.json --device cuda
```

The other two real-model panels use the same argument shape:
`validation.internal_targeted_probe` and
`validation.internal_semantic_robustness`. Give each a different fresh
`--runtime-dir` and a different `--out` path. A failed semantic panel is a
recorded outcome to inspect, not a result to omit.

The corner probe is under `findings/010-corner-audit/probe.py`, preserving its
original relative-path layout. It requires the built checkpoint above and
refuses to overwrite its `isolated-runtime/` directory. Its oracle supplies
expected values for comparison; actual model argmax outputs are fed back.

`ui/server.py --help` documents the optional local renderer and queue bridge.
The bundle contains no running service, authentication configuration, private
network access, or hosted endpoint. Its runtime serializes state separately
from the checkpoint; saved files are needed to continue after process shutdown.

## Scope of redistribution

This is a source snapshot and evidence supplement, not a pretrained model
distribution or a new license grant over upstream material. Third-party terms
remain with their respective projects. No new project-wide software license
has been assigned by preparing this archive.
