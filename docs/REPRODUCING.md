# Checking and reproducing Qwen-Life

There are two different checks here. The small CPU suite verifies the exported
source without loading Qwen. Rebuilding the widened model and running the GPU
validators tests the model itself. Passing the first does not substitute for
the second.

The recorded model results are in `snapshot/reports/`. They include failures,
not just successful runs. The source snapshot was assembled after those
experiments; it is not a commit-exact reconstruction of every historical run.
This publication has not independently repeated the complete GPU experiment.
No pretrained or merged model tensors are included.

## Prepare the pinned source dependencies

Start in the repository root, then enter the immutable source snapshot:

```bash
cd snapshot
mkdir -p references
git clone https://github.com/divagr18/qwenputer.git references/qwenputer
git -C references/qwenputer checkout bec1a0c9654ac8151b6f36ed0154a19c53141384
git clone https://github.com/vixhal-baraiya/qwenGPU.git references/qwenGPU
git -C references/qwenGPU checkout e2695f6a0a818ebc68e2e14b6b990eddf04c51b3
docker build -t qwen-life:dev .
```

These explicit source revisions matter: qwenGPU's package metadata points at a different qwenputer
revision. Follow the checkouts above rather than installing that dependency
indirectly.

The Dockerfile uses NVIDIA's PyTorch `25.10-py3` image and Transformers `5.22.1`.
The recorded GPU environment used Torch `2.9.0a0+145a3a7bda.nv25.10` and CUDA
13.0. The image is selected by tag, and several dependencies have lower version
bounds; this is **not a fully locked environment**. Record your installed
versions when comparing results. `Dockerfile.fast` describes an optional
convolution kernel setup; it removes an incompatible FLA package and does not
establish that FLA acceleration was active.

## Run the small CPU checks

From the `snapshot/` directory used above, return to the repository root and
run the wrapper in the container. This command does not request GPU access:

```bash
cd ..
docker run --rm -v "$PWD:/study" -w /study qwen-life:dev \
  python scripts/run_cpu_checks.py --output-dir checks/cpu-new
```

The wrapper explicitly disables CUDA, uses one CPU thread per numerical
library, and avoids writing Python caches into the snapshot. It runs:

- **8 reference tests**, including every one of the 512 possible binary local
  neighborhoods and comparisons of a small analytic neural network to the
  independent Life rule.
- **31 circuit, merge-plan, and runtime unit tests**. These exercise symbolic
  circuit execution and runtime behavior without loading a pretrained or
  merged checkpoint. Pytest versions with subtest reporting also show nine
  passing subtests.
- **13 UI/bridge unit tests** for the queue, state contract, and interface
  behavior. These do not launch a browser or execute the real model.

`checks/cpu-new/` receives the exact combined output logs and `report.json`,
including versions, dependency revisions, test counts, and exit codes. The
wrapper refuses to overwrite a nonempty output directory. Choose a new name
for another run. A failed test or unexpected number of passed tests makes the
wrapper exit unsuccessfully.

You can also run it with an existing compatible Python environment:

```bash
python3 scripts/run_cpu_checks.py \
  --python /path/to/your/venv/bin/python \
  --qwenputer /path/to/pinned/qwenputer \
  --qwengpu /path/to/pinned/qwenGPU \
  --output-dir checks/cpu-local-new
```

The Python environment needs Torch, Transformers `5.22.1`, pytest, NumPy,
safetensors, and tokenizers. Both dependency directories must be clean Git
checkouts at the pinned revisions. These CPU results are kept separate from
the historical GPU reports.

## Obtain the base model and construct the added weights

Return to `snapshot/`. Download the exact base-model revision using the Python
Hugging Face client already available in the container:

```bash
cd snapshot
docker run --rm -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17", local_dir="base-qwen")'

docker run --rm --ipc=host -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -m life.internal_merge base-qwen build/qwen-life-internal-08b-v3
```

The base model remains subject to its own terms. The merge constructs weights
on the CPU and requires substantial system memory and disk space. It produces
a model with 4,094,404,032 parameters and twelve added layers. It does not run
a gradient-training loop. Use a fresh build destination; keep generated
weights and runtime state out of source control.

## Validate the real merged model

These steps require a compatible NVIDIA GPU, drivers, container GPU support,
and sufficient free memory. Use a separate runtime directory for each panel.
The validators must not point at an interactive world's saved state.

Run the following from `snapshot/` after the build finishes:

```bash
docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -m validation.internal_e2e build/qwen-life-internal-08b-v3 \
  --runtime-dir runs/new-e2e --out results/new-e2e.json --device cuda

docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -m validation.internal_targeted_probe build/qwen-life-internal-08b-v3 \
  --runtime-dir runs/new-targeted --out results/new-targeted.json --device cuda

docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python -m validation.internal_semantic_robustness build/qwen-life-internal-08b-v3 \
  --runtime-dir runs/new-semantic --out results/new-semantic.json --device cuda
```

Inspect the JSON results even if a command exits successfully. In particular,
the semantic and targeted validators record per-case failures in JSON without
necessarily returning a nonzero process exit code:

```bash
python3 - <<'PY'
import json
from pathlib import Path
for name in ("new-e2e", "new-targeted", "new-semantic"):
    report = json.loads((Path("results") / f"{name}.json").read_text())
    print(name, "status:", report.get("status"))
    print(json.dumps(report.get("summary", {}), indent=2))
    for case in report.get("cases", []):
        if case.get("passed") is False:
            print("FAILED CASE:", json.dumps(case, indent=2))
PY
```

The historical core panel passed seven scenarios and 36 isolated protocol
predictions; the targeted panel passed 13 cases. The broader semantic panel
passed **26 of 31**, and its report correctly says **failed**. Keep any new
failures visible. Do not modify the historical reports or relax the expected
actions to make a rerun appear successful.

## Reproduce the 160-generation evolution audit

This probe uses the checkpoint path built above. It creates its own isolated
runtime and refuses to reuse that directory. Run it from `snapshot/`:

```bash
docker run --rm --gpus all --ipc=host -v "$PWD:/workspace/qwen-life" \
  -w /workspace/qwen-life qwen-life:dev \
  python findings/010-corner-audit/probe.py

python3 - <<'PY'
import json
from pathlib import Path
p = Path("findings/010-corner-audit/report.json")
r = json.loads(p.read_text())
print("status:", r["status"])
for case in r["cases"]:
    print(case["name"], case["passed"], case["checked_generations"], case["checked_tokens"])
print("generations:", sum(c["checked_generations"] for c in r["cases"]))
print("protocol outputs:", sum(c["checked_tokens"] for c in r["cases"]))
PY
```

The three worlds run for 64, 64, and 32 generations. The recorded run checked
160 resulting boards and 2,746 internal outputs. The independent Life program
supplies expected values for comparison; the model's actual predicted tokens
are fed back to advance the recurrent state. The probe also records two
language interactions, which are observations rather than additional scored
semantic passes.

The new result is `snapshot/findings/010-corner-audit/report.json`. The historical
record remains separate at `snapshot/reports/010-corner-audit/report.json`.
Use another fresh checkout for another audit run, or deliberately archive your
own generated runtime and report before rerunning. Never overwrite the stored
historical evidence to substitute a new result.

## What these checks establish

The exhaustive 512-neighborhood check applies to the small local rule and
standalone analytic network. The symbolic circuit tests check a different
layer of the implementation. The real-checkpoint validators test selected
model executions. None proves correctness for every possible 16-by-16 world,
every conversation, or indefinitely many generations.

Historical live-browser reports are included, but the original browser replay
script, which contained a private server address, is omitted from the source
snapshot. The CPU UI tests are reproducible here; the historical browser
scenarios require a separately configured live interface and browser harness.

The early three-layer model and later twelve-layer internal controller are
different checkpoints. Keep their evidence and timing measurements separate.
The complete GPU build and validation sequence above remains an instruction
for reproduction, not a claim that the publication process reran it.
