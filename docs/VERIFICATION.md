# How we verified it

We checked the rules, the model's execution, and its interpretation of requests
separately. A correct Life circuit can still receive the wrong command.

## 1. Check the local rule

A square and its eight neighbors have 512 possible on/off arrangements. The
[standalone tests](../snapshot/reference/test_life.py) enumerate every one and
compare a small analytically constructed neural network with the Life rule.
This exhaustively checks that local construction. It is not exhaustive
validation of the expanded Qwen checkpoint.

## 2. Check the actual model

The [evolution probe](../snapshot/findings/010-corner-audit/probe.py) runs the
expanded Qwen checkpoint from three starting boards: a block with a blinker,
a glider away from the diagonal, and a seeded random board. The runs cover
64, 64, and 32 generations.

A separate Life program calculates expected results. At each output step the
probe compares the model's highest-scoring token with the expected token, then
feeds the **model's own token** into the next call. It also reads and checks
the recurrent board after every complete generation. Expected answers are
used for comparison, not substituted for model predictions.

The [saved report](../snapshot/reports/010-corner-audit/report.json) records
**160 matching generations and 2,746 matching output tokens**. This is evidence
for those runs, not a proof for every starting board, every conversation, or
unlimited execution.

## 3. Check whether language chooses the intended action

The [semantic panel](../snapshot/validation/internal_semantic_robustness.py)
compares the resulting board with the intended effect of each request. It
includes supported operations, requests to leave the board alone, ordinary
language, and repeated requests in a conversation.

Its [report](../snapshot/reports/004-internal-controller/semantic-v3.json)
records **26 passes and five failures** and retains `status: "failed"`. The
failures are `block_lower_right`, `glider_upper_left`, `repeat_setup_clear`,
`repeat_setup_blinker`, and `block_same_prompt_first`. The separate 13-case
targeted panel does not replace this broader result.

Read the JSON outcome when rerunning a validator: the semantic script can
finish with exit code zero while its report contains failed cases.

## 4. Check persistence and the interface separately

The [core report](../snapshot/reports/004-internal-controller/e2e-v3.json) and
[restart report](../snapshot/reports/004-internal-controller/service-restart-v3.json)
record saving and restoring the world. The historical
[live-browser report](../snapshot/reports/004-internal-controller/ui-artifacts-v3/report.json)
records 11 passing scenarios. The original browser replay script is not in
this package. UI fixtures and bridge unit tests are not evidence of neural
execution, even when they pass.

## 5. Check this publication package

On 16 November 2026 we reran the selected CPU tests from the preserved snapshot:
8 standalone Life tests, 31 circuit/merge/runtime tests with 9 subtests, and
13 UI bridge tests. **All 52 tests and 9 subtests passed.** The
[machine-readable report](../checks/cpu-2026-11-16/report.json) and adjacent
logs record exactly what ran and the local dependency versions.

These new checks used the existing local environment with the GPU disabled;
they did not rebuild the full checkpoint or repeat the GPU and browser runs.
The snapshot manifest preserves the original source hashes and lists report
redactions. Source is a dated snapshot, not a commit-exact reconstruction of
every historical run. A complete independent reproduction remains outstanding.

See [REPRODUCING.md](REPRODUCING.md) for commands and
[the original supplement notes](../snapshot/README.md) for historical scope.
