# ENR00 source correspondence and differences

Prepared 2026-10-05. Scope: local code and historical configuration metadata.

- Canonical source remains under the original experiment `src/oasst1_filter`
  tree. No source file in that tree was changed for release preparation.
- The candidate follows 27 existing Python files (1,327,886 bytes) reachable by
  static local imports from the four CF12/R16 responsibility owners. It also
  includes two existing behavior-test files and the original pyproject.
- Static closure includes function-local imports, but dynamic external assets
  and code extracted/launched by the judge still require backend acceptance.
- The inspected formal freezes bind configurations and assets. They do not
  contain a complete directly usable historical source-file hash manifest.
- The separate R16 preformal `READINESS.json` binds four source files. Current
  `arms/paired_filtering.py`, `train/run_codecontests.py` and
  `runtime/nat_spec.py` match its hashes. Current `analysis/codecontests.py`
  does not: readiness `a9faec5ac0c5b7200579575102b5dea5ae065b3a0c4e7bc6f2c3525dabad83eb`,
  current `39177dfc6f7e704aece63883e7587fc98a6db1504ee27a5e9cb5464ba753c348`.
- Current analysis contains later asset-analysis responsibilities. The original
  full source snapshot was not restored for a line-by-line comparison. A
  passing synthetic suite does not prove that every historical implementation
  detail remained unchanged. Describe the release as the current implementation
  supporting the documented experiments, not a byte-identical historical
  execution snapshot or a reconstructed experimental Git commit.
- The 22 selected existing tests passed under the specified local WSL runtime.
  No model, GPU, candidate evaluation or scientific result recalculation was
  performed for this publication task.
- The original package's NumPy pin differs from the recorded cloud runtime.
  README and RUN_PROFILES disclose this rather than silently rewriting the
  dependency record.

Internal detailed evidence is in PREPARATION_CHECK.json and RELEASE_MANIFEST.csv.
Those internal records retain complete local paths and must not be uploaded
unreviewed. Original protocols, freezes, raw plans and private helpers require
separate publication review; they are excluded from the candidate archive.
