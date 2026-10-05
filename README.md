# Conditional training utility of G121: CF12 and R16 experiment code

Code publication prepared on 2026-10-05 for the public repository
[fanfoxle/g121-conditional-utility](https://github.com/fanfoxle/g121-conditional-utility).
First-party code is licensed under MIT. No real author name or personal email
is added. Use the actual publication commit as the code version; it is not a
historical experimental commit. No release tag or archive DOI is asserted.

## Scope

The code supports a fixed natural-record subgroup experiment for code generation.
CF12 compared B, B+G, B+H and B+W in 12 paired training contexts. R16 subsequently
compared B and B+G in 16 new paired initialization/training contexts. G contains
121 records, selected by the previously fixed complementary token-balancing
rule. The same HumanEval+164 panel was reused across research decisions.

Reported R16 results are conditional exploratory group-policy evidence. They
do not establish intrinsic data quality, individual harmful records, an isolated
error mechanism, or independent external confirmation. R16's original terminal
is `POSITIVE_BELOW_CONTINUATION_SCALE`; its predefined 2 percentage-point
continuation criterion and stop decision remain in force.

This code publication contains the existing stable implementation and its
required local import closure. Some retained modules also support historical
experiments because the current entry point imports them. They are not separate
experiments to run. No implementation tree was copied or refactored for release.

## Implementation and reported versions

All implementation files retain their original paths under
`20_q1_operational_defect_audit/experiment/src/oasst1_filter/`.

| Responsibility | Existing module |
| --- | --- |
| Fixed exposures, blocks and subgroup plans | `arms/paired_filtering.py` |
| Shared initial states, token-weighted training, optimizer and recovery | `train/run_codecontests.py`, `train/stream.py` |
| Generation, fixed HumanEval+ interface and isolated scoring | `eval/humaneval.py`, `eval/code_exec.py` |
| Fixed readouts and terminal decisions | `analysis/codecontests.py` |
| CF12/R16 preparation, workers, budget guardian and publication | `runtime/nat_spec.py` |

`RUN_PROFILES.json` contains selected historical configuration and seed metadata.
It is documentation, not a replacement for the original design and execution
receipts. `SOURCE_RELATION.md` states exactly what source-version correspondence
has been verified. The release commit, once created, identifies the published
code snapshot; it must not be presented as a historical experimental commit.

## Environment

Formal runtime metadata records Python 3.10.12, torch 2.9.1+cu130,
transformers 5.3.0, peft 0.18.1 and NumPy 2.1.2. The historical target was a
single RTX 5090 under native Linux. Training, GPU generation and global CPU
scoring concurrency were 2/4/4, with training and GPU generation mutually
exclusive. CPU scoring can overlap subsequent GPU work.

Native Linux namespace/chroot/seccomp support and the required permissions are
necessary for the existing scoring path. WSL was used for local CPU development.
Candidate programs must run through the existing isolated judge. Windows alone
does not provide the formal backend.

Additional imported packages include SciPy, PyArrow and EvalPlus. Their complete
historical distribution versions have not been established by this release
inspection. EvalPlus **code revision** is
`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`; HumanEvalPlus **test asset** version
is v0.1.10. The asset version is not a software-package version.

The original `pyproject.toml` pins NumPy 2.2.6 and SciPy 1.15.3 for the local
package. It is not the full historical cloud dependency lock and its NumPy
pin differs from the recorded formal environment. Preserve the formal recipe
when provisioning a reproduction environment rather than silently using the
package metadata as a historical lock. Establish an explicit compatible CUDA
runtime and remaining dependencies before attempting a new GPU run.

## Required external materials

The repository contains code and documentation. It excludes dataset payloads,
model or adapter weights, credentials, private cloud helpers, personal paths,
full execution receipts, generated completions, paper files and caches.

Obtain upstream resources from their publishers, subject to their terms:

- CodeContests: `deepmind/code_contests`, train split, revision
  `802411c3010cb00d1b05bad57ca77365a3c699d6` at
  <https://huggingface.co/datasets/deepmind/code_contests>.
- Qwen2.5-1.5B base revision
  `8faed761d45a263340a0528343f099c05c9a4323` at
  <https://huggingface.co/Qwen/Qwen2.5-1.5B>.
- EvalPlus source at the specified revision:
  <https://github.com/evalplus/evalplus/tree/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e>.
  Obtain the matching HumanEvalPlus v0.1.10 asset through its publisher;
  `RUN_PROFILES.json` retains the expected asset and archive identities.

Exact historical rerunning additionally needs project-derived G membership,
the B exposure ledger, token IDs/loss masks/canonical blocks, design receipts,
historical seed exclusions and accepted machine configuration. Downloading
upstream datasets alone does not recreate those bindings. Distribution or an
external access channel for these derived materials has not been authorized;
no public download URL is asserted. Their internal locations and prerequisites
are listed in `MATERIALS.md`. This limits independent end-to-end reproduction
until an access route or documented portable reconstruction is supplied.

## Checks that can be run without research assets

From the repository root, in an environment with Python 3.10, pytest, NumPy
and SciPy, run only the two existing synthetic behavior suites:

```sh
mkdir -p .release-test-tmp
cd 20_q1_operational_defect_audit/experiment
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_q1_w259_cf12.py tests/test_q1_g121_replication.py \
  --basetemp=../../.release-test-tmp/pytest
```

These tests check plan, state identity, scheduling, budget, analysis-boundary
and publication behavior using synthetic fixtures. They neither reproduce
model outcomes nor establish GPU/environment portability. During release
preparation, 22 tests passed using the existing local WSL runtime: Python
3.10.12, pytest 9.0.3, NumPy 2.2.6 and SciPy 1.15.3.

## Historical experiment entry points and reproduction procedure

The following steps describe real existing interfaces; they are not automatic
commands to reopen the closed studies.

1. Recover the fixed derived assets and corresponding source/design identities.
   Keep all paths in a workspace, with datasets and models obtained separately.
2. Build plans using `build_w259_factorial_plans` in `arms/paired_filtering.py`.
   The existing `prepare_w259_cf12_design` and `prepare_g121_design` functions
   in `runtime/nat_spec.py` bind the original assets and exclusions. These
   functions retain project-specific historical path and digest requirements.
3. Complete machine acceptance and fresh paired initial-state checks on the
   actual backend. Do not substitute synthetic trained adapters for initial
   states. CF12 uses `prepare_w259_cf12_execution`; R16 uses
   `prepare_g121_execution` and its protocol/run-bound authorization record.
4. Use a new output directory. Never overwrite historical terminal outputs.
   CF12 uses the `w259-cf12-run` command without `--authorization-path`;
   R16 uses the same command with that argument. The fixed design selects
   the scientific protocol; a directory name alone does not select it.
5. The existing workers carry out training, generation and isolated scoring,
   followed by the fixed readout and five-file scientific delivery. Keep the
   original 100 updates, normalization, seeds, endpoint and stopping rules.

R16 command shape, **after** valid design, acceptance and authorization records
have been prepared in a separate reproduction workspace:

```sh
export PYTHONPATH="$PWD/20_q1_operational_defect_audit/experiment/src"
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m oasst1_filter.runtime.nat_spec w259-cf12-run \
  --task-root "$PWD/.scratch/q1-g121-r16" \
  --output-root "$PWD/20_q1_operational_defect_audit/experiment/artifacts/q1_g121_r16/REPRODUCTION_RUN" \
  --authorization-path "$PWD/.scratch/q1-g121-r16/AUTHORIZATION.json"
```

The existing CLI also exposes `w259-cf12-prepare` and `w259-cf12-worker`.
`w259-cf12-prepare` is the CF12 preparation path; it must not be substituted
for R16's different preparation function. A fresh reproduction must resolve
the embedded historical bindings and resource permissions deliberately.

## Publication, licensing and citation

First-party code and documentation use the MIT License in `LICENSE`, selected
under the author's delegated instruction to follow comparable research code.
Public examples include [LESS](https://github.com/princeton-nlp/LESS/blob/main/LICENSE)
and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/LICENSE).
EvalPlus uses a different license; third-party code/data/model terms remain
separate. This candidate does not redistribute EvalPlus source or its tests.
No real author name, personal email, ORCID or fabricated citation metadata is
provided. The generic contributors notice does not identify individual authors.

For citation, use this repository title, the visible repository account
`fanfoxle` (not a verified personal author name), publication year 2026 and
the actual immutable commit URL. No release tag or archive DOI is asserted.
Code availability is not a newly completed independent replication.
