# Qwen-Life: source and verification evidence

This repository accompanies **Programming a persistent Game of Life world into
Qwen**. The experiment installs a directly constructed Life circuit into added
Qwen layers. Fixed weights implement the rules; recurrent state carries the
16 × 16 board. The surrounding software saves state, feeds model outputs back,
and displays the result.

The purpose of this package is to make the implementation and its evidence
inspectable and provide instructions for rerunning it. **It is not a claim of
an independently reproduced result.**

## Start here

- [How we verified it](docs/VERIFICATION.md): the checks, results, and limits.
- [Reproduction instructions](docs/REPRODUCING.md): dependencies, CPU checks,
  checkpoint construction, and GPU validators.
- [Recorded CPU rerun](checks/cpu-2026-11-16/report.json): **52 tests and 9 subtests
  passed** on 16 November 2026. These checks did not load the full checkpoint.
- [Source snapshot](snapshot/): the preserved 7 November 2026 supplement.
- [Snapshot provenance and checksums](snapshot/MANIFEST.json).

The source snapshot and the historical reports are kept together, unchanged
from that supplement. Its README records the original scope and environment.
The current reproduction instructions and new CPU-check logs are outside it,
so they cannot be mistaken for historical experiment outputs.

## What the evidence says

| Check | Result | Scope |
| --- | --- | --- |
| Standalone analytic Life network | All 512 local neighborhoods matched | Exhaustive local rule check; not the full Qwen checkpoint |
| Expanded Qwen evolution audit | 160 generations and 2,746 internal output tokens matched | Three starting worlds; model predictions drove execution |
| Language interpretation | 26 of 31 requests passed; five failed | The failed status and failed cases are retained |
| Historical live browser scenarios | 11 of 11 passed | Interface behavior, separate from the circuit checks |
| New CPU rerun | 52 tests and 9 subtests passed | Standalone rule, symbolic circuit, merge/runtime checks, and UI bridge unit tests |

The current internal-controller checkpoint has 4,094,404,032 parameters and
twelve added layers. Reports for the earlier three-layer checkpoint are also
included and clearly labeled; their results should not be pooled with the
current design.

## Verify the files

From this repository's root, using Python 3.11 or later:

```bash
python3 scripts/verify_evidence.py
sha256sum -c SHA256SUMS
```

The first command verifies the preserved snapshot hashes and checks that the
headline numbers agree with the saved reports. The second checks all packaged
files. These are integrity checks, not new model evaluations. See
[REPRODUCING.md](docs/REPRODUCING.md) to execute tests.

## What is established prior work

Hand-engineered neural weights that implement Life already appear in Springer
and Kenyon's 2020 paper,
[It's Hard for Neural Networks to Learn the Game of Life](https://arxiv.org/html/2009.01398v1).
Life without additional training is not the novelty of this experiment.
Our focus is the integration into pretrained Qwen, recurrent board storage,
language-selected commands, and the measured limitations of that combination.
That integration alone does not establish research novelty.

We use [qwenputer](https://github.com/divagr18/qwenputer)'s `chipformer` compiler
and the `counter` and `thermo` helpers distributed with
[qwenGPU](https://github.com/vixhal-baraiya/qwenGPU). The qwenGPU helper source
credits those counters to Qwenputer's RISC-V implementation. Exact revisions
and base-model provenance are in the reproduction guide.

## Boundaries of this release

The reports record earlier local runs. The source snapshot is not a
commit-exact reconstruction of every historical run. The new CPU checks used
the existing local Python environment and pinned upstream checkouts; they do
not validate a fresh Docker build or repeat the GPU experiments. Model weights,
live user state, and the original browser replay script are not included.
Private machine paths and demo URLs in historical reports were replaced by
labeled placeholders, recorded in the snapshot manifest.

No new project-wide software license is assigned by this publication. Base
model and upstream project terms remain with their respective authors.

## Push this prepared repository

The local `origin` is configured for `https://github.com/octaviuslabs/qwen-life.git`.
Once that repository exists and your GitHub credentials are available:

```bash
git push -u origin main
```

No remote content has been changed by preparing this directory. If the remote
already has commits, fetch and reconcile its history before pushing; do not
force-push over it.
