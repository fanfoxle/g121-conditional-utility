"""Pure eight-root CodeContests V1 J1-95 analysis."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import math
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from ..protocol import (
    CODECONTESTS_FORMAL_ROOTS,
    CODECONTESTS_V2_FORMAL_ROOTS,
    CODECONTESTS_V2_PROTOCOL_ID,
    CODECONTESTS_V2_STAGE_ARMS,
)


_QUANTUM = Decimal("0.000000000001")
_MARGIN = Decimal("0.020000000000")


def resolve_g121_terminal(*, complete, mean=None, sd=None, lower=None, upper=None):
    """The fixed replication's ordered descriptive readout; no optional stopping."""
    if not complete:
        return "MEASUREMENT_INVALID"
    if sd == 0:
        return "UNCERTAINTY_DEGENERATE"
    if any(v is None or not math.isfinite(v) for v in (mean,sd,lower,upper)):
        raise ValueError("G121 complete readout lacks finite statistics")
    if upper < 0:
        return "DIRECTION_REVERSED"
    if lower > 0:
        return "POSITIVE_CONDITIONAL_REPLICATION" if mean >= 2 else "POSITIVE_BELOW_CONTINUATION_SCALE"
    return "NO_SUPPORT_AT_CONTINUATION_SCALE" if upper < 2 else "WITHIN_BUDGET_UNRESOLVED"


def analyze_g121_positions(rows, *, pairs, task_ids):
    """Analyze only the new 16 pairs, retaining fixed missing-position denominators."""
    tasks = tuple(task_ids)
    if tasks != tuple(f"HumanEval/{i}" for i in range(164)):
        raise ValueError("G121 requires the frozen 164-task panel")
    if tuple(p["pair_id"] for p in pairs) != tuple(f"R{i:02d}" for i in range(1,17)):
        raise ValueError("G121 requires all sixteen new pairs")
    roots = {p["pair_id"]:p["root"] for p in pairs}
    seeds = {p["pair_id"]:p["init_seed"] for p in pairs}
    if len(set(roots.values()) | set(seeds.values())) != 32:
        raise ValueError("G121 pair seed values must be distinct")
    expected = {(p["pair_id"],a,t) for p in pairs for a in ("B","BG") for t in tasks}
    grid = {}
    for row in rows:
        key = (row.get("pair_id"),row.get("arm"),row.get("task_id"))
        if key not in expected or key in grid or row.get("root") != roots.get(key[0]) or row.get("init_seed") != seeds.get(key[0]):
            raise ValueError("G121 position identity differs")
        y = row.get("Y")
        if y is not None and (type(y) is not int or y not in (0,1)):
            raise ValueError("G121 Y must be binary or missing")
        grid[key] = y
    if set(grid) != expected:
        raise ValueError("G121 must include all 5248 keys, explicitly marking missing Y")
    root_rows = []
    for p in pairs:
        pid = p["pair_id"]
        r = {"pair_id":pid,"init_seed":p["init_seed"],"root":p["root"]}
        for a in ("B","BG"):
            values = [grid[pid,a,t] for t in tasks]
            r[f"{a}_pass"] = sum(y == 1 for y in values)
            r[f"{a}_valid"] = sum(y is not None for y in values)
        paired = [(grid[pid,"B",t],grid[pid,"BG",t]) for t in tasks]
        r["gain"] = sum(a==1 and b==0 for a,b in paired)
        r["loss"] = sum(a==0 and b==1 for a,b in paired)
        r["missing"] = 328-r["B_valid"]-r["BG_valid"]
        r["net_pass"] = r["B_pass"]-r["BG_pass"] if r["missing"] == 0 else None
        r["gross"] = r["gain"]+r["loss"]
        r["d_pp"] = 100*r["net_pass"]/164 if r["net_pass"] is not None else None
        r["bound_lower_pp"] = 100*(r["B_pass"]-r["BG_pass"]-(164-r["BG_valid"]))/164
        r["bound_upper_pp"] = 100*(r["B_pass"]+164-r["B_valid"]-r["BG_pass"])/164
        root_rows.append(r)
    complete = all(r["missing"] == 0 for r in root_rows)
    values = [r["d_pp"] for r in root_rows]
    mean = sum(values)/16 if complete else None
    sd = math.sqrt(sum((v-mean)**2 for v in values)/15) if complete else None
    t95, t975 = 2.131449545559323, 2.489879703472758
    ci = [mean-t95*sd/4,mean+t95*sd/4] if complete and sd > 0 else None
    reference = [mean-t975*sd/4,mean+t975*sd/4] if complete and sd > 0 else None
    return {"complete":complete,"terminal":resolve_g121_terminal(complete=complete,mean=mean,sd=sd,
            lower=ci[0] if ci else None,upper=ci[1] if ci else None),
            "n":16,"valid_positions":sum(y is not None for y in grid.values()),"expected_positions":5248,
            "root_rows":root_rows,"d_pp":values,"mean_pp":mean,"sample_sd_pp":sd,
            "ci95_pp":ci,"reference_97_5_ci_pp":reference,"t95":t95,"t97_5":t975,
            "B_pass":sum(r["B_pass"] for r in root_rows),"BG_pass":sum(r["BG_pass"] for r in root_rows),
            "net_pass":sum(r["net_pass"] for r in root_rows) if complete else None,
            "gain":sum(r["gain"] for r in root_rows),"loss":sum(r["loss"] for r in root_rows),
            "gross":sum(r["gross"] for r in root_rows),
            "positive_pairs":sum(v is not None and v>0 for v in values),
            "zero_pairs":sum(v == 0 for v in values),"negative_pairs":sum(v is not None and v<0 for v in values),
            "missingness_bound_pp":[sum(r[k] for r in root_rows)/16 for k in ("bound_lower_pp","bound_upper_pp")]}


def analyze_g121_crossrun(batches):
    """Describe two complete B/BG grids; pairs never cross batch boundaries.

    The caller binds historical metadata and maps context_id to pair_id.
    This seam accepts strict binary Y only and freezes 164 tasks, 12/16 pairs
    and k=10. It deliberately performs no inferential or historical analysis.
    """
    tasks = tuple(f"HumanEval/{i}" for i in range(164))
    if set(batches) != {"CF12", "R16"}:
        raise ValueError("XRUN_INPUT_INVALID: requires CF12 and R16")
    task_rows = [{"task_id": t} for t in tasks]
    summaries = {}
    for batch, prefix, n in (("CF12", "C", 12), ("R16", "R", 16)):
        pairs = tuple(f"{prefix}{i:02d}" for i in range(1, n + 1))
        expected = {(p, a, t) for p in pairs for a in ("B", "BG") for t in tasks}
        grid = {}
        for row in batches[batch]:
            key = (row.get("pair_id"), row.get("arm"), row.get("task_id"))
            if key not in expected or key in grid:
                raise ValueError("XRUN_INPUT_INVALID: duplicate or foreign position")
            if type(row.get("Y")) is not int or row["Y"] not in (0, 1):
                raise ValueError("XRUN_INPUT_INVALID: Y must be binary integer")
            grid[key] = row["Y"]
        if set(grid) != expected:
            raise ValueError("XRUN_INPUT_INVALID: missing positions")
        for tr in task_rows:
            values = [(grid[p, "B", tr["task_id"]], grid[p, "BG", tr["task_id"]]) for p in pairs]
            gain = values.count((1, 0))
            loss = values.count((0, 1))
            both = values.count((1, 1))
            net, gross = gain - loss, gain + loss
            counts = {"n": n, "B": gain + both, "BG": loss + both,
                      "gain": gain, "loss": loss, "both": both,
                      "neither": values.count((0, 0)), "net": net, "gross": gross,
                      "q": net / n, "c_pp": 100 * net / (n * 164), "t": gross / n,
                      "B_rate": (gain + both) / n, "BG_rate": (loss + both) / n,
                      "sign": "positive" if net > 0 else "negative" if net < 0 else "zero"}
            tr.update({f"{batch}_{k}": v for k, v in counts.items()})
        totals = {k: sum(r[f"{batch}_{k}"] for r in task_rows)
                  for k in ("B", "BG", "gain", "loss", "both", "neither", "net", "gross")}
        concentration = {}
        for kind in ("positive", "negative", "gross"):
            weights = [(r[f"{batch}_gross"] if kind == "gross" else
                        max(r[f"{batch}_net"] * (1 if kind == "positive" else -1), 0), i)
                       for i, r in enumerate(task_rows)]
            weights.sort(key=lambda item: (-item[0], item[1]))
            denominator = sum(w for w, _ in weights)
            numerator = sum(w for w, _ in weights[:10])
            concentration[kind] = {"k": 10, "numerator": numerator, "denominator": denominator,
                                   "nonzero_tasks": sum(w > 0 for w, _ in weights),
                                   "share": numerator / denominator if denominator else None}
        summaries[batch] = {"n": n, "positions": len(grid), "paired_positions": n * 164,
                            "totals": totals, "per_context": {k: v / n for k, v in totals.items()},
                            "mean_pp": 100 * totals["net"] / (n * 164), "concentration": concentration}
    signs = ("positive", "zero", "negative")
    cells = {f"{a}/{b}": 0 for a in signs for b in signs}
    for row in task_rows:
        row["nine_cell"] = f'{row["CF12_sign"]}/{row["R16_sign"]}'
        cells[row["nine_cell"]] += 1
    same = cells["positive/positive"] + cells["negative/negative"]
    opposite = cells["positive/negative"] + cells["negative/positive"]
    both_zero = cells["zero/zero"]
    directions = {k: {"count": v, "proportion_all_tasks": v / 164} for k, v in
                  {"same_nonzero": same, "opposite_nonzero": opposite, "both_zero": both_zero,
                   "one_zero": 164 - same - opposite - both_zero}.items()}
    no_transition = sum(r["CF12_gross"] == r["R16_gross"] == 0 for r in task_rows)
    directions.update(nine_cells=cells, both_nonzero_count=same + opposite,
                      both_nonzero_coverage=(same + opposite) / 164,
                      conditional_agreement=same / (same + opposite) if same + opposite else None,
                      both_zero_no_transitions=no_transition,
                      both_zero_with_cancellation=both_zero - no_transition)
    overlap = {}
    for kind in ("T", "P", "N"):
        sets = [{r["task_id"] for r in task_rows if
                 (r[f"{b}_gross"] > 0 if kind == "T" else
                  r[f"{b}_net"] > 0 if kind == "P" else r[f"{b}_net"] < 0)} for b in ("CF12", "R16")]
        old, new = sets
        intersection, union = len(old & new), len(old | new)
        overlap[kind] = {"CF12_size": len(old), "R16_size": len(new),
                         "intersection": intersection, "union": union,
                         "only_CF12": len(old - new), "only_R16": len(new - old),
                         "jaccard": intersection / union if union else None}
    return {"analysis_id": "q1-g121-xrun-aa-v1", "complete": True,
            "terminal": "COMPLETED_Q1_G121_XRUN_AA", "task_count": 164,
            "task_rows": task_rows, "batches": summaries, "direction": directions, "overlap": overlap}


def publish_g121_crossrun(result, output_dir):
    """Publish the bounded cross-run result and verify decompressed ZIP bytes.

    Report prose is supplied by the task's interpretation step. No source
    result or scientific label is edited. A failed run never fabricates rows.
    """
    import csv
    import io
    import json
    from pathlib import Path
    import zipfile

    rows = result.get("task_rows", [])
    complete = result.get("complete") is True
    if complete and [r.get("task_id") for r in rows] != [f"HumanEval/{i}" for i in range(164)]:
        raise ValueError("XRUN_PUBLICATION_INVALID: requires all 164 ordered tasks")
    report = result.get("report_markdown")
    if not isinstance(report, str) or not report.strip():
        raise ValueError("XRUN_PUBLICATION_INVALID: report is required")
    output = Path(output_dir)
    if not complete and (output / "TASK_COMPARISON.csv").exists():
        raise ValueError("XRUN_PUBLICATION_INVALID: refusing stale success table")
    published = {k: v for k, v in result.items() if k not in {"task_rows", "report_markdown"}}
    payloads = {"RESULT.json": (json.dumps(published, ensure_ascii=False, indent=2,
                 sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
                "REPORT.md": report.encode("utf-8")}
    if complete:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        payloads["TASK_COMPARISON.csv"] = buffer.getvalue().encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (output / name).write_bytes(data)
    with zipfile.ZipFile(output / "DELIVERY.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payloads[name])
    with zipfile.ZipFile(output / "DELIVERY.zip") as archive:
        if len(archive.namelist()) != len(payloads) or set(archive.namelist()) != set(payloads):
            raise ValueError("XRUN_PUBLICATION_INVALID: ZIP membership")
        if any(archive.read(name) != (output / name).read_bytes() for name in payloads):
            raise ValueError("XRUN_PUBLICATION_INVALID: decompressed bytes differ")
    return hashlib.sha256((output / "DELIVERY.zip").read_bytes()).hexdigest()


def analyze_cf12_assets(rows, *, contexts, task_ids):
    """Describe the complete frozen CF12 grid; no new inferential decisions.

    Inputs contain only binary outcomes and identity metadata. The existing
    CF12 analyzer supplies historical statistics, with the same frozen rules.
    Counts remain integers; percentage points use the original 164 denominator.
    """
    if any(type(row.get("Y")) is not int or row["Y"] not in (0, 1) for row in rows):
        raise ValueError("AA_INPUT_INVALID: Y must be a strict binary integer")
    if any("state_id" in r and r["state_id"] != f'{r.get("context_id")}_{r.get("arm")}' for r in rows):
        raise ValueError("AA_INPUT_INVALID: state_id differs from context/arm")
    original = analyze_w259_cf12_positions(rows, contexts=contexts, task_ids=task_ids)
    arms = ("B", "BG", "BH", "BW")
    comparisons = {"hG": ("B", "BG"), "hH": ("B", "BH"),
                   "cG": ("BH", "BW"), "cH": ("BG", "BW"), "hW": ("B", "BW")}
    counts = ("gain", "loss", "both", "neither", "net", "gross")
    tasks = sorted(task_ids)
    grid = {(r["context_id"], r["task_id"], r["arm"]): r["Y"] for r in rows}
    task_rows = {t: {"task_id": t, **{f"{n}_{k}": 0 for n in comparisons for k in counts},
                     "J_net": 0} for t in tasks}
    context_rows = []
    patterns = {t: {f"{i:04b}": 0 for i in range(16)} for t in tasks}
    joint = {f"{a},{b}": 0 for a in (-1, 0, 1) for b in (-1, 0, 1)}
    for context in contexts:
        cid = context["context_id"]
        cr = {"context_id": cid, "root": context["root"],
              "init_seed": context.get("init_seed", "unknown"),
              **{f"{a}_pass": 0 for a in arms},
              **{f"{n}_{k}": 0 for n in comparisons for k in counts}, "J_net": 0}
        for t in tasks:
            y = {a: grid[cid, t, a] for a in arms}
            patterns[t]["".join(str(y[a]) for a in arms)] += 1
            for a in arms:
                cr[f"{a}_pass"] += y[a]
            d = {}
            for n, (a, c) in comparisons.items():
                gain, loss = int(y[a] == 1 and y[c] == 0), int(y[a] == 0 and y[c] == 1)
                values = (gain, loss, y[a] * y[c], (1-y[a]) * (1-y[c]), gain-loss, gain+loss)
                d[n] = gain-loss
                for k, v in zip(counts, values):
                    cr[f"{n}_{k}"] += v
                    task_rows[t][f"{n}_{k}"] += v
            interaction = y["BG"] + y["BH"] - y["B"] - y["BW"]
            if not interaction == d["hW"]-d["hG"]-d["hH"] == d["cG"]-d["hG"] == d["cH"]-d["hH"]:
                raise ValueError("AA_VALIDATION_FAILED: position J identity")
            cr["J_net"] += interaction
            task_rows[t]["J_net"] += interaction
            joint[f'{d["hG"]},{d["hH"]}'] += 1
        for n in (*comparisons, "J"):
            cr[f"{n}_pp"] = 100 * cr[f"{n}_net"] / 164
        context_rows.append(cr)
    totals = {n: {k: sum(r[f"{n}_{k}"] for r in context_rows) for k in counts} for n in comparisons}
    concentration = {}
    for n in comparisons:
        nets = [r[f"{n}_net"] for r in task_rows.values()]
        entry = {"positive_tasks": sum(d > 0 for d in nets), "zero_tasks": sum(d == 0 for d in nets),
                 "negative_tasks": sum(d < 0 for d in nets), "P": sum(max(d, 0) for d in nets),
                 "N": sum(max(-d, 0) for d in nets), "net": sum(nets)}
        for kind in ("positive", "negative", "absolute", "gross"):
            weights = {t: (max(task_rows[t][f"{n}_net"], 0) if kind == "positive" else
                           max(-task_rows[t][f"{n}_net"], 0) if kind == "negative" else
                           abs(task_rows[t][f"{n}_net"]) if kind == "absolute" else
                           task_rows[t][f"{n}_gross"]) for t in tasks}
            order = sorted(tasks, key=lambda t: (-weights[t], t))
            denominator = sum(weights.values())
            entry[kind] = {"denominator": denominator, "top_k": {
                str(k): {"task_ids": order[:k], "numerator": sum(weights[t] for t in order[:k]),
                         "share": sum(weights[t] for t in order[:k])/denominator if denominator else None}
                for k in (1, 5, 10)}}
        concentration[n] = entry
    footprints, overlap = {}, {}
    for kind in ("positive", "negative", "transition"):
        sets = {n: {t for t in tasks if (task_rows[t][f"{n}_net"] > 0 if kind == "positive" else
                                       task_rows[t][f"{n}_net"] < 0 if kind == "negative" else
                                       task_rows[t][f"{n}_gross"] > 0)} for n in ("hG", "hH", "hW")}
        footprints[kind] = {n: sorted(s) for n, s in sets.items()}
        overlap[kind] = {}
        for a, b in (("hG", "hH"), ("hG", "hW"), ("hH", "hW")):
            intersection, union = sets[a] & sets[b], sets[a] | sets[b]
            overlap[kind][f"{a}:{b}"] = {"intersection": len(intersection), "union": len(union),
                                        "jaccard": len(intersection)/len(union) if union else None}
    heterogeneity = {}
    for n in (*comparisons, "J"):
        values = [r[f"{n}_pp"] for r in context_rows]
        facts = original["contrasts"][n]
        heterogeneity[n] = {k: facts[k] for k in ("mean_pp", "sample_sd_pp", "positive_count", "zero_count",
                                                  "negative_count", "leave_one_context_mean_range_pp")}
        heterogeneity[n].update(min_pp=min(values), max_pp=max(values))
    return {"analysis_id": "q1-cf12-aa-v1", "complete": True, "terminal": "COMPLETED_Q1_CF12_AA",
            "original_analysis": original, "position_count": len(rows), "task_count": 164, "context_count": 12,
            "arm_totals": {a: sum(r[f"{a}_pass"] for r in context_rows) for a in arms},
            "comparisons": totals, "J_net": sum(r["J_net"] for r in context_rows),
            "task_rows": list(task_rows.values()), "context_rows": context_rows,
            "pattern_rows": [{"task_id": t, "pattern": p, "count": count}
                             for t in tasks for p, count in patterns[t].items()],
            "pattern_order": list(arms), "pattern_totals": {p: sum(patterns[t][p] for t in tasks) for p in patterns[tasks[0]]},
            "concentration": concentration, "footprints": footprints, "footprint_overlap": overlap,
            "joint_hG_hH": {"counts": joint, "same_direction": joint["1,1"] + joint["-1,-1"],
                            "opposite_direction": joint["1,-1"] + joint["-1,1"],
                            "one_sided": sum(joint[f"{a},{b}"] for a, b in ((0,1),(0,-1),(1,0),(-1,0))),
                            "both_zero": joint["0,0"], "opposite_cells_are_structural_zeros": True},
            "heterogeneity": heterogeneity}


def publish_cf12_assets(result, output_dir):
    """Publish the finite AA result, or an honest two-file failure delivery."""
    import csv
    import json
    from pathlib import Path
    import zipfile

    output = Path(output_dir)
    complete = result.get("complete") is True
    tables = {"TASK_TABLE.csv": result.get("task_rows", []),
              "CONTEXT_TABLE.csv": result.get("context_rows", []),
              "PATTERN_TABLE.csv": result.get("pattern_rows", [])}
    if complete and [len(v) for v in tables.values()] != [164, 12, 2624]:
        raise ValueError("AA_VALIDATION_FAILED: publication table sizes")
    output.mkdir(parents=True, exist_ok=True)
    names = ["REPORT.md", "RESULT.json"]
    if complete:
        for name, rows in tables.items():
            with (output / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            names.append(name)
    else:
        # Never leave stale success tables beside an incomplete result.
        for name in tables:
            (output / name).unlink(missing_ok=True)
    published = {k: v for k, v in result.items() if k not in {"task_rows", "context_rows", "pattern_rows"}}
    (output / "RESULT.json").write_text(json.dumps(published, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Q1-CF12-AA", "", f'Terminal: {result["terminal"]}; complete={complete}.', "",
             "Post-hoc/descriptive analysis of published binary Y only. No new scientific gate, p-value, interval or panel.",
             "", "## Source binding", "", "```json", json.dumps(result.get("source_sha256", {}), indent=2, sort_keys=True), "```",
             "", "## Execution", "", "```json", json.dumps(result.get("execution", {}), indent=2, sort_keys=True), "```", ""]
    if complete:
        lines += ["## 1. Reconstruction", "", f'7,872 unique positions; 164 tasks; 12 joint (init_seed, root) contexts. Arm passes: {result["arm_totals"]}.',
                  f'Original terminal remains {result["original_analysis"]["terminal"]}. Historical means, sample SD and working intervals reproduced within 1e-8 pp.',
                  "", "## 2. Directional transitions", "", "Comparison | gain | loss | both | neither | net | gross", "---|---:|---:|---:|---:|---:|---:"]
        for n, v in result["comparisons"].items():
            lines.append(n + " | " + " | ".join(str(v[k]) for k in ("gain", "loss", "both", "neither", "net", "gross")))
        lines += ["", "Gain/loss mean directional pass transitions, not knowledge acquisition/forgetting.", "", "## 3. All-task concentration", "",
                  "All 164 tasks retained. Shares use positive P, negative N, absolute or gross denominators; zero denominators are null. Ties use lexical task_id.", ""]
        for n, v in result["concentration"].items():
            shares = {kind: {k: item["share"] for k, item in v[kind]["top_k"].items()} for kind in ("positive", "negative", "absolute", "gross")}
            lines.append(f'- {n}: positive/zero/negative={v["positive_tasks"]}/{v["zero_tasks"]}/{v["negative_tasks"]}; P={v["P"]}, N={v["N"]}, net={v["net"]}; top-1/5/10 shares={json.dumps(shares, sort_keys=True)}.')
        lines += ["", "## 4. Footprints and 16 patterns", "", "```json", json.dumps(result["footprint_overlap"], indent=2, sort_keys=True), "```",
                  f'Joint hG/hH transitions: {json.dumps(result["joint_hG_hH"], sort_keys=True)}.',
                  "Opposite hG/hH cells are structural zeros because B is shared and binary; this is not evidence of a common mechanism.",
                  "Pattern bit order B/BG/BH/BW; all 16 patterns including zeros are reported for every task (12 observations each).",
                  f'Global pattern counts: {json.dumps(result["pattern_totals"], sort_keys=True)}.', "", "## 5. All-context heterogeneity", "",
                  "All 12 contexts are retained, including C02/C10. Joint init/root variation cannot identify either cause separately.", "",
                  "Contrast | mean pp | sample SD pp | min pp | max pp | positive/zero/negative | leave-one mean range pp", "---|---:|---:|---:|---:|---|---"]
        for n, v in result["heterogeneity"].items():
            lines.append(f'{n} | {v["mean_pp"]:.9f} | {v["sample_sd_pp"]:.9f} | {v["min_pp"]:.9f} | {v["max_pp"]:.9f} | {v["positive_count"]}/{v["zero_count"]}/{v["negative_count"]} | {v["leave_one_context_mean_range_pp"]}')
        lines += ["", f'J total={result["J_net"]}; J=hW-hG-hH=cG-hG=cH-hH is checked at each position, task and context. Crossing zero does not establish no interaction.',
                  "", "## 6. Evidence limits and open questions", "",
                  "The original corrected intervals and labels are retained in RESULT.json.original_analysis. H unresolved does not establish harmlessness; G passing does not establish G is worse than H.",
                  "Net concentration and gross conversion concentration answer different questions: cross-context cancellation can hide conversions. The full task table exposes both; overlap does not establish a shared program mechanism.",
                  "The working-interval units remain 12 training contexts. The 1,968 paired positions and 164 tasks add no independent training repetitions. HE adaptive reuse remains; no independent confirmation or training-record attribution is possible.",
                  "Open questions concern stability outside this reused endpoint, separation of initialization/root variability, and whether observed net contributions persist. These assets cannot answer them, identify harmful training records, or guarantee recursive splitting will work. No successor experiment is designed or launched."]
        old = result["original_analysis"]["contrasts"]
        if all(old[n]["bonferroni_97_5_ci_pp"] is not None for n in ("hG", "hH")):
            lines += ["", f'The corrected lower endpoints are hG={old["hG"]["bonferroni_97_5_ci_pp"][0]:.9f} pp and hH={old["hH"]["bonferroni_97_5_ci_pp"][0]:.9f} pp. Interpret their gate difference alongside the reported means and context SDs, not as a ranking of harm.']
        for n in ("hG", "hH", "hW"):
            con = result["concentration"][n]
            pos = con["positive"]["top_k"]["10"]["share"]
            gross = con["gross"]["top_k"]["10"]["share"]
            if pos is not None and gross is not None:
                lines.append(f'{n}: the top ten tasks account for {pos:.2%} of positive net contribution but {gross:.2%} of gross transitions. Both dispersed transitions and net cancellation matter; no tasks are excluded.')
    else:
        lines += [f'Reason: {result.get("reason", "unknown")}', f'Completed items: {result.get("completed_items", [])}', "No complete scientific tables are claimed."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "DELIVERY.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 18, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (output / name).read_bytes())
    digest = hashlib.sha256((output / "DELIVERY.zip").read_bytes()).hexdigest()
    (output / "DELIVERY.zip.sha256").write_text(digest + "  DELIVERY.zip\n", encoding="ascii")
    return names + ["DELIVERY.zip", "DELIVERY.zip.sha256"]


def analyze_nat_lbpp_fixed16_positions(
    *,
    position_rows: Sequence[Mapping[str, object]],
    roots: Sequence[int],
    task_ids: Sequence[str],
) -> dict[str, object]:
    """Compute the frozen finite-panel F-minus-M estimand and deletion envelope."""

    if len(roots) != 8 or len(set(roots)) != 8:
        raise ValueError("NAT-LBPP-FIX16 requires eight unique roots")
    if len(task_ids) != 161 or len(set(task_ids)) != 161:
        raise ValueError("NAT-LBPP-FIX16 requires 161 unique tasks")
    expected = {
        (int(root), arm, str(task_id))
        for root in roots
        for arm in ("POL_FILTER", "POL_MSEP")
        for task_id in task_ids
    }
    observed: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for row in position_rows:
        key = (int(row["root"]), str(row["arm"]), str(row["task_id"]))
        if key not in expected:
            raise ValueError("NAT-LBPP-FIX16 position key is foreign")
        if key in observed:
            raise ValueError("NAT-LBPP-FIX16 position key is duplicated")
        if row.get("state_id") != f"{key[0]}_{key[1]}" or row.get("Y") not in (None, 0, 1):
            raise ValueError("NAT-LBPP-FIX16 position identity or outcome is invalid")
        observed[key] = row
    if set(observed) != expected:
        raise ValueError("NAT-LBPP-FIX16 position panel is missing frozen keys")

    root_rows: list[dict[str, object]] = []
    total_f = total_m = valid_f = valid_m = 0
    pair_counts = {"positive": 0, "flat": 0, "negative": 0}
    for root in roots:
        counts: dict[str, tuple[int, int]] = {}
        for arm in ("POL_FILTER", "POL_MSEP"):
            values = [observed[(int(root), arm, str(task_id))]["Y"] for task_id in task_ids]
            counts[arm] = (
                sum(int(value) for value in values if value in (0, 1)),
                sum(value in (0, 1) for value in values),
            )
        f_pass, f_valid = counts["POL_FILTER"]
        m_pass, m_valid = counts["POL_MSEP"]
        net = f_pass - m_pass
        pair_counts["positive" if net > 0 else "negative" if net < 0 else "flat"] += 1
        root_rows.append({
            "root": int(root), "F_pass": f_pass, "F_valid": f_valid,
            "M_pass": m_pass, "M_valid": m_valid, "net": net,
            "d_r": net / 161 if f_valid == m_valid == 161 else None,
        })
        total_f += f_pass
        total_m += m_pass
        valid_f += f_valid
        valid_m += m_valid

    denominator = 8 * 161
    missing_f, missing_m = denominator - valid_f, denominator - valid_m
    lower = Fraction(total_f - total_m - missing_m, denominator)
    upper = Fraction(total_f + missing_f - total_m, denominator)
    complete = valid_f == valid_m == denominator
    total_net = total_f - total_m
    theta = Fraction(total_net, denominator) if complete else None
    leave_one = (
        [Fraction(total_net - int(row["net"]), 7 * 161) for row in root_rows]
        if complete else []
    )
    envelope = [min([theta, *leave_one]), max([theta, *leave_one])] if complete else [None, None]
    terminal = None
    if complete:
        if total_net >= 26 and all(value > 0 for value in leave_one):
            terminal = "NAT_LBPP_FIX16_POSITIVE_FINITE_PANEL_SUPPORT"
        elif total_net <= -26 and all(value < 0 for value in leave_one):
            terminal = "NAT_LBPP_FIX16_ADVERSE_FINITE_PANEL_CONTRAST"
        elif envelope[0] > Fraction(-1, 50) and envelope[1] < Fraction(1, 50):
            terminal = "NAT_LBPP_FIX16_BELOW_INVESTMENT_SCALE"
        else:
            terminal = "NAT_LBPP_FIX16_DECISION_UNRESOLVED"
    floor = total_f == total_m == 0 and complete
    ceiling = total_f == total_m == denominator and complete
    return {
        "complete": complete,
        "valid_positions": valid_f + valid_m,
        "F_pass": total_f, "F_valid": valid_f,
        "M_pass": total_m, "M_valid": valid_m,
        "total_net": total_net if complete else None,
        "theta": float(theta) if theta is not None else None,
        "root_rows": root_rows,
        "pair_counts": pair_counts,
        "leave_one_root_out": [float(value) for value in leave_one],
        "J": [float(value) if value is not None else None for value in envelope],
        "jackknife_envelope": [float(value) if value is not None else None for value in envelope],
        "missingness_bound": [float(lower), float(upper)],
        "terminal": terminal,
        "information_flag": "FLOOR_UNINFORMATIVE" if floor else "CEILING_UNINFORMATIVE" if ceiling else None,
    }


def nat_msep_missingness_bound(
    *,
    f_pass: int,
    f_valid: int,
    m_pass: int,
    m_valid: int,
    denominator: int = 164,
) -> tuple[Fraction, Fraction]:
    """Return sharp paired-effect bounds when invalid endpoints remain missing.

    Missing F observations can only increase the contrast and missing M
    observations can only decrease it.  Fractions keep the gate exact.
    """

    values = (f_pass, f_valid, m_pass, m_valid, denominator)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("counts and denominator must be integers")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if not (0 <= f_pass <= f_valid <= denominator):
        raise ValueError("F counts must satisfy 0 <= pass <= valid <= denominator")
    if not (0 <= m_pass <= m_valid <= denominator):
        raise ValueError("M counts must satisfy 0 <= pass <= valid <= denominator")
    f_missing = denominator - f_valid
    m_missing = denominator - m_valid
    return (
        Fraction(f_pass - m_pass - m_missing, denominator),
        Fraction(f_pass + f_missing - m_pass, denominator),
    )


def resolve_nat_onoff_terminal(
    *,
    complete: bool,
    all_zero_or_full: bool = False,
    mean: float | None = None,
    sd: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    measurement_invalid: bool = False,
    safety_or_permission: bool = False,
    scientific_review: bool = False,
    resource_incomplete: bool = False,
    infrastructure_unrecoverable: bool = False,
) -> str:
    """Apply the frozen Q1-NAT-ONOFF-354 terminal precedence literally."""

    if measurement_invalid:
        return "Q1_ONOFF_MEASUREMENT_INVALID"
    if safety_or_permission:
        return "Q1_ONOFF_BLOCKED_SAFETY_OR_PERMISSION"
    if scientific_review:
        return "Q1_ONOFF_NEEDS_SCIENTIFIC_REVIEW"
    if resource_incomplete:
        return "Q1_ONOFF_INCOMPLETE_RESOURCE"
    if infrastructure_unrecoverable:
        return "Q1_ONOFF_INCOMPLETE_INFRASTRUCTURE"
    if not complete:
        raise ValueError("incomplete Q1-NAT-ONOFF analysis needs a stopping condition")
    if all_zero_or_full:
        return "Q1_ONOFF_READOUT_DEGENERATE"
    if None in (mean, sd, lower, upper):
        raise ValueError("complete Q1-NAT-ONOFF analysis needs root statistics")
    assert mean is not None and sd is not None and lower is not None and upper is not None
    if sd > 0 and lower > 0 and mean >= 2.0:
        return "Q1_ONOFF_POSITIVE_EXPLORATORY"
    if sd > 0 and upper < 2.0:
        return "Q1_ONOFF_BELOW_INVESTMENT_SCALE"
    return "Q1_ONOFF_UNRESOLVED_WITHIN_BUDGET"


def analyze_nat_onoff_positions(
    rows: Sequence[Mapping[str, object]],
    *,
    roots: Sequence[int],
    task_ids: Sequence[str],
) -> dict[str, object]:
    """Summarize paired OFF-minus-ON HumanEval+ pass effects at root level."""

    root_order = tuple(roots)
    task_order = tuple(task_ids)
    if len(root_order) != 8 or len(set(root_order)) != 8 or not task_order:
        raise ValueError("Q1-NAT-ONOFF analysis geometry is incomplete")
    expected = {
        (root, arm, task_id)
        for root in root_order
        for arm in ("ON", "OFF")
        for task_id in task_order
    }
    table: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (int(row["root"]), str(row["arm"]), str(row["task_id"]))
        if key in table:
            raise ValueError("Q1-NAT-ONOFF position key is duplicated")
        table[key] = row
    if set(table) != expected:
        raise ValueError("Q1-NAT-ONOFF position table does not close its geometry")
    denominator = len(task_order)
    root_rows: list[dict[str, object]] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    root_d: list[float] = []
    for root in root_order:
        counts: dict[str, tuple[int, int]] = {}
        for arm in ("OFF", "ON"):
            values = [table[(root, arm, task_id)].get("Y") for task_id in task_order]
            if any(value not in (0, 1, None) for value in values):
                raise ValueError("Q1-NAT-ONOFF Y must be 0, 1, or missing")
            counts[arm] = (sum(value == 1 for value in values), sum(value in (0, 1) for value in values))
        off_pass, off_valid = counts["OFF"]
        on_pass, on_valid = counts["ON"]
        missing_off, missing_on = denominator - off_valid, denominator - on_valid
        bound_lower = 100.0 * (off_pass - on_pass - missing_on) / denominator
        bound_upper = 100.0 * (off_pass + missing_off - on_pass) / denominator
        lower_values.append(bound_lower)
        upper_values.append(bound_upper)
        delta = (
            100.0 * (off_pass - on_pass) / denominator
            if off_valid == on_valid == denominator
            else None
        )
        if delta is not None:
            root_d.append(delta)
        root_rows.append(
            {
                "root": root,
                "OFF_pass": off_pass,
                "OFF_valid": off_valid,
                "ON_pass": on_pass,
                "ON_valid": on_valid,
                "d_pp": delta,
                "bound_lower_pp": bound_lower,
                "bound_upper_pp": bound_upper,
                "missing_OFF": missing_off,
                "missing_ON": missing_on,
            }
        )
    complete = len(root_d) == 8
    mean = sum(root_d) / 8 if complete else None
    sd = (
        math.sqrt(sum((value - mean) ** 2 for value in root_d) / 7)
        if complete and mean is not None
        else None
    )
    ci = (
        [mean - 2.364624251 * sd / math.sqrt(8), mean + 2.364624251 * sd / math.sqrt(8)]
        if complete and mean is not None and sd is not None
        else [None, None]
    )
    leave_one = (
        [(sum(root_d) - value) / 7 for value in root_d] if complete else []
    )
    all_counts = [row[f"{arm}_pass"] for row in root_rows for arm in ("ON", "OFF")]
    degenerate = complete and (
        all(value == 0 for value in all_counts)
        or all(value == denominator for value in all_counts)
    )
    terminal = (
        resolve_nat_onoff_terminal(
            complete=True,
            all_zero_or_full=degenerate,
            mean=mean,
            sd=sd,
            lower=ci[0],
            upper=ci[1],
        )
        if complete
        else None
    )
    return {
        "complete": complete,
        "denominator": denominator,
        "root_rows": root_rows,
        "root_d_pp": root_d,
        "mean_d_pp": mean,
        "sd_d_pp": sd,
        "ci95_pp": ci,
        "leave_one_root_mean_range_pp": [min(leave_one), max(leave_one)] if leave_one else [None, None],
        "missingness_bound_pp": [sum(lower_values) / 8, sum(upper_values) / 8],
        "all_zero_or_full": degenerate,
        "terminal": terminal,
    }


def resolve_nat_e95_delete_terminal(
    *,
    complete: bool,
    mean: float | None = None,
    sd: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> str | None:
    """Apply the complete-data E95 deletion classification in frozen order."""

    if not complete:
        return None
    if mean is None or sd is None:
        raise ValueError("complete Q1-NAT-E95-DELETE analysis needs root statistics")
    if sd > 0:
        if lower is None or upper is None:
            raise ValueError("nonconstant Q1-NAT-E95-DELETE effects need an interval")
        if lower > 0 and mean >= 2.0:
            return "E95_DIRECTIONAL_SUPPORT_EXPLORATORY"
        if upper < 0 and mean <= -2.0:
            return "E95_DIRECTION_REVERSED_EXPLORATORY"
    return "E95_SMALL_OR_UNRESOLVED"


def analyze_nat_e95_delete_positions(
    rows: Sequence[Mapping[str, object]],
    *,
    roots: Sequence[int],
    task_ids: Sequence[str],
) -> dict[str, object]:
    """Pair new MID positions with immutable historical ON/OFF endpoints."""

    root_order = tuple(roots)
    task_order = tuple(task_ids)
    if len(root_order) != 8 or len(set(root_order)) != 8 or not task_order:
        raise ValueError("Q1-NAT-E95-DELETE analysis geometry is incomplete")
    expected = {
        (root, arm, task_id)
        for root in root_order
        for arm in ("OFF", "ON", "MID")
        for task_id in task_order
    }
    table: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (int(row["root"]), str(row["arm"]), str(row["task_id"]))
        if key in table:
            raise ValueError("Q1-NAT-E95-DELETE position key is duplicated")
        table[key] = row
    if set(table) != expected:
        raise ValueError("Q1-NAT-E95-DELETE position table does not close its geometry")
    denominator = len(task_order)
    root_rows: list[dict[str, object]] = []
    root_de: list[float] = []
    root_dw: list[float] = []
    de_lower_values: list[float] = []
    de_upper_values: list[float] = []
    dw_lower_values: list[float] = []
    dw_upper_values: list[float] = []
    transitions = {
        "ON0_MID0": 0,
        "ON0_MID1": 0,
        "ON1_MID0": 0,
        "ON1_MID1": 0,
    }
    mid_pass_counts: list[int] = []
    for root in root_order:
        values = {
            arm: [table[(root, arm, task)].get("Y") for task in task_order]
            for arm in ("OFF", "ON", "MID")
        }
        if any(value not in (0, 1) for arm in ("OFF", "ON") for value in values[arm]):
            raise ValueError("historical Q1-NAT-ONOFF endpoints must be complete")
        if any(value not in (0, 1, None) for value in values["MID"]):
            raise ValueError("Q1-NAT-E95-DELETE MID Y must be 0, 1, or missing")
        off_pass = sum(value == 1 for value in values["OFF"])
        on_pass = sum(value == 1 for value in values["ON"])
        mid_pass = sum(value == 1 for value in values["MID"])
        mid_valid = sum(value in (0, 1) for value in values["MID"])
        mid_missing = denominator - mid_valid
        mid_pass_counts.append(mid_pass)
        de_lower = 100.0 * (mid_pass - on_pass) / denominator
        de_upper = 100.0 * (mid_pass + mid_missing - on_pass) / denominator
        dw_lower = 100.0 * (off_pass - mid_pass - mid_missing) / denominator
        dw_upper = 100.0 * (off_pass - mid_pass) / denominator
        de_lower_values.append(de_lower)
        de_upper_values.append(de_upper)
        dw_lower_values.append(dw_lower)
        dw_upper_values.append(dw_upper)
        de = 100.0 * (mid_pass - on_pass) / denominator if mid_valid == denominator else None
        dw = 100.0 * (off_pass - mid_pass) / denominator if mid_valid == denominator else None
        historical = 100.0 * (off_pass - on_pass) / denominator
        identity_error = de + dw - historical if de is not None and dw is not None else None
        if de is not None and dw is not None:
            root_de.append(de)
            root_dw.append(dw)
        for on, mid in zip(values["ON"], values["MID"], strict=True):
            if mid in (0, 1):
                transitions[f"ON{on}_MID{mid}"] += 1
        root_rows.append({
            "root": root,
            "OFF_pass": off_pass,
            "OFF_valid": denominator,
            "ON_pass": on_pass,
            "ON_valid": denominator,
            "MID_pass": mid_pass,
            "MID_valid": mid_valid,
            "missing_MID": mid_missing,
            "dE_pp": de,
            "dW_pp": dw,
            "historical_total_pp": historical,
            "identity_error_pp": identity_error,
            "dE_bound_lower_pp": de_lower,
            "dE_bound_upper_pp": de_upper,
            "dW_bound_lower_pp": dw_lower,
            "dW_bound_upper_pp": dw_upper,
        })
    complete = len(root_de) == 8
    mean = sum(root_de) / 8 if complete else None
    sd = (
        math.sqrt(sum((value - mean) ** 2 for value in root_de) / 7)
        if complete and mean is not None
        else None
    )
    ci = (
        [mean - 2.364624251 * sd / math.sqrt(8), mean + 2.364624251 * sd / math.sqrt(8)]
        if complete and mean is not None and sd is not None and sd > 0
        else [None, None]
    )
    leave_one = [(sum(root_de) - value) / 7 for value in root_de] if complete else []
    terminal = resolve_nat_e95_delete_terminal(
        complete=complete,
        mean=mean,
        sd=sd,
        lower=ci[0],
        upper=ci[1],
    )
    extreme = None
    if complete and all(value == 0 for value in mid_pass_counts):
        extreme = "all_zero"
    elif complete and all(value == denominator for value in mid_pass_counts):
        extreme = "all_full"
    flags = {
        "small_point_effect": bool(complete and mean is not None and abs(mean) < 2.0),
        "direction_unresolved": bool(
            complete
            and (sd == 0 or (ci[0] is not None and ci[1] is not None and ci[0] <= 0 <= ci[1]))
        ),
        "interval_below_investment_scale": bool(
            complete and ci[1] is not None and ci[1] < 2.0
        ),
    }
    return {
        "complete": complete,
        "denominator": denominator,
        "root_rows": root_rows,
        "root_dE_pp": root_de,
        "root_dW_pp": root_dw,
        "mean_dE_pp": mean,
        "sd_dE_pp": sd,
        "ci95_dE_pp": ci,
        "leave_one_root_mean_dE_range_pp": (
            [min(leave_one), max(leave_one)] if leave_one else [None, None]
        ),
        "missingness_bound_dE_pp": [
            sum(de_lower_values) / 8,
            sum(de_upper_values) / 8,
        ],
        "missingness_bound_dW_pp": [
            sum(dw_lower_values) / 8,
            sum(dw_upper_values) / 8,
        ],
        "transition_counts": transitions,
        "extreme_readout": extreme,
        "descriptive_flags": flags,
        "terminal": terminal,
    }


def resolve_nat_msep_formal_terminal(
    *,
    complete: bool,
    measurement_invalid: bool = False,
    safety_or_permission: bool = False,
    root_set_qualified: bool = True,
    hard_cap_reached: bool = False,
    infrastructure_unrecoverable: bool = False,
    root_sd: float | None = None,
    mean: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> str:
    """Resolve the protocol's closed terminal set in precedence order."""

    if measurement_invalid:
        return "NAT_MSEP_20_MEASUREMENT_INVALID"
    if safety_or_permission:
        return "NAT_MSEP_20_STOPPED_SAFETY_OR_PERMISSION"
    if not root_set_qualified:
        return "NAT_MSEP_20_FORMAL_ROOT_SET_NOT_QUALIFIED"
    if hard_cap_reached and not complete:
        return "NAT_MSEP_20_INCOMPLETE_RESOURCE_CAP"
    if infrastructure_unrecoverable and not complete:
        return "NAT_MSEP_20_INCOMPLETE_INFRASTRUCTURE"
    if not complete:
        raise ValueError("an incomplete run needs a declared stopping condition")
    statistics = (root_sd, mean, lower, upper)
    if any(value is None or not math.isfinite(float(value)) for value in statistics):
        raise ValueError("a complete run requires finite root_sd, mean, lower, and upper")
    if float(root_sd) < 0.0 or float(lower) > float(upper):
        raise ValueError("invalid complete-run statistics")
    if float(root_sd) == 0.0:
        return "NAT_MSEP_20_INFORMATION_INSUFFICIENT"
    if float(upper) < 0.0:
        return "NAT_MSEP_20_ACTIVE_CONTRAST_ADVERSE"
    if float(upper) < 0.02:
        return "NAT_MSEP_20_BELOW_INVESTMENT_SCALE"
    if float(mean) >= 0.02 and float(lower) > 0.0:
        return "NAT_MSEP_20_POSITIVE_EXPLORATORY"
    return "NAT_MSEP_20_UNRESOLVED_WITHIN_BUDGET"


def analyze_nat_msep_positions(
    *, position_rows: Sequence[Mapping[str, object]], roots: Sequence[int]
) -> dict[str, object]:
    """Validate the frozen 8x2x164 panel and compute equal-root F-minus-M statistics."""

    if len(roots) != 8 or len(set(roots)) != 8:
        raise ValueError("NAT-MSEP analysis requires eight unique roots")
    expected = {
        (int(root), arm, f"HumanEval/{index}")
        for root in roots
        for arm in ("POL_FILTER", "POL_MSEP")
        for index in range(164)
    }
    observed: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for row in position_rows:
        key = (int(row["root"]), str(row["arm"]), str(row["task_id"]))
        if key not in expected or key in observed:
            raise ValueError("NAT-MSEP position key is foreign or duplicated")
        if row.get("state_id") != f"{key[0]}_{key[1]}" or row.get("Y") not in (
            None,
            0,
            1,
        ):
            raise ValueError("NAT-MSEP position identity or outcome is invalid")
        observed[key] = row
    if set(observed) != expected:
        raise ValueError("NAT-MSEP position panel is not the frozen 2,624 keys")

    root_rows: list[dict[str, object]] = []
    deltas: list[float] = []
    complete = True
    for root in roots:
        counts: dict[str, tuple[int, int]] = {}
        for arm in ("POL_FILTER", "POL_MSEP"):
            values = [
                observed[(int(root), arm, f"HumanEval/{index}")].get("Y")
                for index in range(164)
            ]
            counts[arm] = (
                sum(int(value) for value in values if value in (0, 1)),
                sum(value in (0, 1) for value in values),
            )
        f_pass, f_valid = counts["POL_FILTER"]
        m_pass, m_valid = counts["POL_MSEP"]
        lower, upper = nat_msep_missingness_bound(
            f_pass=f_pass, f_valid=f_valid, m_pass=m_pass, m_valid=m_valid
        )
        valid_root = f_valid == m_valid == 164
        complete = complete and valid_root
        delta = Fraction(f_pass - m_pass, 164) if valid_root else None
        if delta is not None:
            deltas.append(float(delta))
        root_rows.append(
            {
                "root": int(root),
                "F_pass": f_pass,
                "F_valid": f_valid,
                "M_pass": m_pass,
                "M_valid": m_valid,
                "F_minus_M": f_pass - m_pass if valid_root else None,
                "delta": float(delta) if delta is not None else None,
                "bound_lower": float(lower),
                "bound_upper": float(upper),
                "missing_F": 164 - f_valid,
                "missing_M": 164 - m_valid,
            }
        )
    if complete:
        mean = sum(deltas) / 8
        variance = sum((value - mean) ** 2 for value in deltas) / 7
        root_sd = math.sqrt(variance)
        half_width = 2.364624251 * root_sd / math.sqrt(8)
        interval: list[float | None] = [mean - half_width, mean + half_width]
        terminal = resolve_nat_msep_formal_terminal(
            complete=True,
            root_sd=root_sd,
            mean=mean,
            lower=interval[0],
            upper=interval[1],
        )
    else:
        mean = root_sd = None
        interval = [None, None]
        terminal = None
    return {
        "complete": complete,
        "valid_positions": sum(row.get("Y") in (0, 1) for row in position_rows),
        "root_rows": root_rows,
        "mean": mean,
        "root_sd": root_sd,
        "working_interval": interval,
        "terminal": terminal,
    }


def nat_asset_numeric_summary(values):
    """Finite observations only, explicit missing/invalid coverage, type-7 quantiles."""
    valid, missing, invalid = [], 0, 0
    for value in values:
        if value is None or value in ("", "null"):
            missing += 1
            continue
        try:
            number = float(value)
            if isinstance(value, bool) or not math.isfinite(number):
                raise ValueError
            valid.append(number)
        except (ValueError, TypeError):
            invalid += 1
    valid.sort()
    def quantile(p):
        if not valid:
            return None
        index = (len(valid)-1)*p
        lower = int(index)
        return valid[lower] + (valid[min(lower+1,len(valid)-1)]-valid[lower])*(index-lower)
    return dict(total=len(values), valid=len(valid), missing=missing, invalid=invalid,
        coverage=len(valid)/len(values) if values else 0,
        mean=sum(valid)/len(valid) if valid else None,
        quartiles=[quantile(p) for p in (.25,.5,.75)], min=min(valid) if valid else None,
        max=max(valid) if valid else None)


def analyze_nat_asset_training(*, updates, terminal, unit_metadata, expected_updates=100, eos_token_id=None):
    """Describe already-recorded exposure and curves; missing assets remain missing."""
    import hashlib
    import json
    from collections import Counter, defaultdict
    result=dict(exposure=dict(status="unavailable",reason="actual plan unavailable"),
                curves=dict(status="unavailable",reason="training terminal unavailable"),unit_rows=[])
    if updates is not None:
        groups, unit_tokens, unit_counts = {}, Counter(), Counter()
        denominators, component_bins, b80_hashes = [], defaultdict(list), []
        components=sorted({e["component"] for update in updates for e in update})
        partial_ratios, sequences, prompts, eos_mismatches = [], [], [], 0
        for update in updates:
            counts=Counter()
            for e in update:
                q=e["retained_response_tokens"]
                if len(e["token_ids"])!=len(e["loss_mask"]) or sum(e["loss_mask"][1:])!=q or q<=0:
                    raise ValueError("NAT-AA shift denominator differs from recorded quota")
                unit,component=e["unit_id"],e["component"]
                meta=unit_metadata.get(unit,{})
                key=(component,unit)
                if key not in groups:
                    groups[key]=dict(component=component,unit_id=unit,problem_id=meta.get("problem_id"),
                        full_exposures=0,partial_exposures=0,retained_tokens=0,original_response_tokens=meta.get("response_tokens"))
                groups[key]["full_exposures" if e["complete"] else "partial_exposures"]+=1
                groups[key]["retained_tokens"]+=q
                unit_tokens[unit]+=q
                unit_counts[unit]+=1
                counts[component]+=q
                sequences.append(len(e["token_ids"]))
                prompts.append(len(e["token_ids"])-q)
                if not e["complete"]:
                    original=meta.get("response_tokens")
                    partial_ratios.append(q/original if original else None)
                if e["complete"] and eos_token_id is not None and e["token_ids"][-1]!=eos_token_id:
                    eos_mismatches+=1
            total=sum(counts.values())
            if not total:
                raise ValueError("NAT-AA empty update denominator")
            denominators.append(total)
            for c in components:
                component_bins[c].append(counts[c])
            b80_hashes.append(hashlib.sha256(json.dumps([e for e in update if e["component"]=="B80"],sort_keys=True,separators=(",",":")).encode()).hexdigest())
        total=sum(unit_tokens.values())
        component_totals={c:sum(qs) for c,qs in component_bins.items()}
        for row in groups.values():
            row["component_token_share"]=row["retained_tokens"]/component_totals[row["component"]]
            row["arm_token_share"]=row["retained_tokens"]/total
        result["unit_rows"]=list(groups.values())
        result["exposure"]=dict(status="available",retained_tokens=total,unique_units=len(unit_tokens),
            unique_problems=len({unit_metadata[u]["problem_id"] for u in unit_tokens if unit_metadata.get(u,{}).get("problem_id")}),
            problem_identity_coverage=sum(bool(unit_metadata.get(u,{}).get("problem_id")) for u in unit_tokens)/len(unit_tokens),
            max_unit_token_share=max(unit_tokens.values())/total,ESS=total**2/sum(q*q for q in unit_tokens.values()),
            component_totals=component_totals,components={c:dict(unique_units=sum(row["component"]==c for row in groups.values()),
                unique_problems=len({row["problem_id"] for row in groups.values() if row["component"]==c and row["problem_id"] is not None}),
                full_exposures=sum(row["full_exposures"] for row in groups.values() if row["component"]==c),
                partial_exposures=sum(row["partial_exposures"] for row in groups.values() if row["component"]==c),
                ESS=1/sum(row["component_token_share"]**2 for row in groups.values() if row["component"]==c),
                max_unit_token_share=max(row["component_token_share"] for row in groups.values() if row["component"]==c)) for c in components},
            full_exposures=sum(row["full_exposures"] for row in groups.values()),partial_exposures=sum(row["partial_exposures"] for row in groups.values()),
            partial_retained_ratios=partial_ratios,partial_ratio_summary=nat_asset_numeric_summary(partial_ratios),
            unit_repeat_distribution=dict(Counter(unit_counts.values())),unit_repeat_summary=nat_asset_numeric_summary(list(unit_counts.values())),
            sequence_lengths=nat_asset_numeric_summary(sequences),prompt_lengths=nat_asset_numeric_summary(prompts),
            complete_EOS_mismatches=eos_mismatches if eos_token_id is not None else None,
            update_denominators=denominators,update_denominator_summary=nat_asset_numeric_summary(denominators),
            component_update_tokens=dict(component_bins),
            update_averaged_nominal_loss_share={c:sum(q/d for q,d in zip(qs,denominators))/len(denominators) for c,qs in component_bins.items()},
            component_update_share_summary={c:nat_asset_numeric_summary([q/d for q,d in zip(qs,denominators)]) for c,qs in component_bins.items()},
            B80_update_identity=b80_hashes,observed_updates=len(updates),expected_updates=expected_updates)
    if terminal is not None:
        telemetry=terminal.get("telemetry",{})
        curves={}
        for key in ("losses","update_seconds"):
            raw=telemetry.get(key,[])
            if not isinstance(raw,list):
                raw=[]
            values=raw[:expected_updates]+[None]*max(0,expected_updates-len(raw))
            clean=[]
            for v in values:
                try:
                    clean.append(float(v) if v is not None and math.isfinite(float(v)) else None)
                except (TypeError,ValueError):
                    clean.append(None)
            curves[key]=clean
            curves["loss_summary" if key=="losses" else "update_seconds_summary"]=nat_asset_numeric_summary(values)
            curves[key+"_extra_entries"]=max(0,len(raw)-expected_updates)
        curves.update(status="available",completed_updates=terminal.get("completed_updates"),
            first10_loss_mean=nat_asset_numeric_summary(curves["losses"][:10])["mean"],
            last10_loss_mean=nat_asset_numeric_summary(curves["losses"][-10:])["mean"],
            gradient_norm="not recorded",validation_loss="not recorded")
        result["curves"]=curves
    return result


def analyze_nat_asset_tables(rows, *, roots, task_ids):
    """Post-hoc full-panel descriptions; no outcome selection or new inference gate."""
    from collections import Counter
    if not roots or len(set(roots)) != len(roots) or len(task_ids) < 2 or len(set(task_ids)) != len(task_ids):
        raise ValueError("NAT-AA invalid fixed roots/tasks")
    states = ["BASE"] + [f"{r}_{a}" for r in roots for a in ("POL_KEEP","POL_FILTER")]
    expected = {(s,t) for s in states for t in task_ids}
    table = {}
    for raw in rows:
        key = (raw["state_id"],raw["task_id"])
        if key in table:
            raise ValueError("NAT-AA duplicate position")
        if key not in expected or raw["Y"] not in (0,1,"0","1") or isinstance(raw["Y"],bool):
            raise ValueError("NAT-AA foreign position or invalid outcome")
        table[key] = dict(raw, Y=int(raw["Y"]))
    if set(table) != expected:
        raise ValueError("NAT-AA missing fixed positions")
    n, nr = len(task_ids), len(roots)
    y = lambda s,t: table[s,t]["Y"]
    categories = sorted({str(v.get("technical_status", "missing")) for v in table.values()})
    root_rows, transitions = [], {}
    for r in roots:
        k,f = f"{r}_POL_KEEP",f"{r}_POL_FILTER"
        cells = Counter(f"{y('BASE',t)}{y(k,t)}{y(f,t)}" for t in task_ids)
        row = dict(root=r,denominator=n,**{f"cell_{i:03b}":cells[f"{i:03b}"] for i in range(8)})
        for label,state in (("K",k),("F",f)):
            gain = sum(y("BASE",t)==0 and y(state,t)==1 for t in task_ids)
            loss = sum(y("BASE",t)==1 and y(state,t)==0 for t in task_ids)
            row.update({f"{label}_gain":gain,f"{label}_loss":loss,f"{label}_net_vs_BASE":gain-loss,
                        f"{label}_passed":sum(y(state,t) for t in task_ids)})
            for b in (0,1):
                denominator = sum(y("BASE",t)==b for t in task_ids)
                row[f"BASE{b}_denominator"] = denominator
                row[f"{label}_pass_rate_given_BASE{b}"] = sum(y(state,t) for t in task_ids if y("BASE",t)==b)/denominator if denominator else None
        row.update(new_pass_contribution=row["F_gain"]-row["K_gain"],retention_contribution=row["K_loss"]-row["F_loss"])
        row["net"] = row["F_passed"]-row["K_passed"]
        row["delta"] = row["net"]/n
        assert row["net"] == row["new_pass_contribution"]+row["retention_contribution"]
        root_rows.append(row)
        matrices = {}
        for name,left,right in (("K_to_F",k,f),("BASE_to_K","BASE",k),("BASE_to_F","BASE",f)):
            counts = Counter((str(table[left,t].get("technical_status","missing")),str(table[right,t].get("technical_status","missing"))) for t in task_ids)
            matrices[name] = {f"{a} -> {b}":counts[a,b] for a in categories for b in categories}
        transitions[str(r)] = matrices
    task_rows = []
    for t in sorted(task_ids,key=lambda t:int(t.split("/")[-1])):
        ks,fs = [[y(f"{r}_{a}",t) for r in roots] for a in ("POL_KEEP","POL_FILTER")]
        positive,negative = sum(f>k for k,f in zip(ks,fs)),sum(f<k for k,f in zip(ks,fs))
        task_rows.append(dict(task_id=t,BASE=y("BASE",t),K=ks,F=fs,net=sum(fs)-sum(ks),
            positive_roots=positive,negative_roots=negative,F_only=positive,K_only=negative,
            any_disagreement=bool(positive+negative),all16_pass=all(ks+fs),all16_fail=not any(ks+fs),
            BASE_category=table["BASE",t].get("technical_status"),BASE_length=table["BASE",t].get("generated_tokens"),
            K_categories=[table[f"{r}_POL_KEEP",t].get("technical_status") for r in roots],
            F_categories=[table[f"{r}_POL_FILTER",t].get("technical_status") for r in roots],
            K_lengths=[table[f"{r}_POL_KEEP",t].get("generated_tokens") for r in roots],
            F_lengths=[table[f"{r}_POL_FILTER",t].get("generated_tokens") for r in roots]))
    total = sum(r["net"] for r in root_rows)
    mean = total/(nr*n)
    ds = [r["delta"] for r in root_rows]
    sd = float(np.std(ds,ddof=1)) if nr>1 else None
    main = dict(delta=mean,delta_pp=mean*100,sample_sd=sd,SE=sd/math.sqrt(nr) if sd is not None else None)
    if nr==8 and n==285:
        original = summarize_nat_replace(list(table.values()),roots=roots,task_ids=task_ids)
        main.update(ci95=original["ci95"],scientific_disposition=original["scientific_disposition"])
    kr,fr = [[r[f"{a}_passed"]/n for r in root_rows] for a in ("K","F")]
    variability = dict(K_sample_sd=float(np.std(kr,ddof=1)) if nr>1 else None,
        F_sample_sd=float(np.std(fr,ddof=1)) if nr>1 else None,
        paired_covariance=float(np.cov(kr,fr,ddof=1)[0,1]) if nr>1 else None)
    ranking = sorted(task_rows,key=lambda t:(-abs(t["net"]),int(t["task_id"].split("/")[-1])))
    absolute = sum(abs(t["net"]) for t in task_rows)
    concentration = dict(absolute_net_sum=absolute,top1_share=abs(ranking[0]["net"])/absolute if absolute else None,
        top5_share=sum(abs(t["net"]) for t in ranking[:5])/absolute if absolute else None,
        ranking=[dict(task_id=t["task_id"],net=t["net"]) for t in ranking])
    def sensitivity(items):
        if not items:
            return dict(range=None,max_influence_positions=[])
        influence=max(abs(v-mean) for _,v in items)
        return dict(range=[min(v for _,v in items),max(v for _,v in items)],
                    max_absolute_change=influence,max_influence_positions=[key for key,v in items if abs(abs(v-mean)-influence)<1e-15],evaluations=len(items))
    loo = dict(label="POST_HOC_SENSITIVITY",leave_one_root_out=sensitivity([(r["root"],(total-r["net"])/((nr-1)*n)) for r in root_rows] if nr>1 else []),
        leave_one_task_out=sensitivity([(t["task_id"],(total-t["net"])/(nr*(n-1))) for t in task_rows]),task_out_denominator=n-1)
    def length_summary(items):
        valid,missing,invalid=[],0,0
        for item in items:
            v=item.get("generated_tokens")
            if v is None or v in ("","null"):
                missing+=1
            elif (type(v) is int or isinstance(v,str) and v.isdigit()) and 0<=int(v)<=1024:
                valid.append(int(v))
            else:
                invalid+=1
        result=nat_asset_numeric_summary(valid)
        result.update(total=len(items),valid=len(valid),missing=missing,invalid=invalid,coverage=len(valid)/len(items) if items else 0,
            bins=[sum(lo<=v<=hi for v in valid) for lo,hi in ((0,64),(65,128),(129,256),(257,512),(513,1023),(1024,1024))])
        assert sum(result["bins"])+missing+invalid==len(items)
        return result
    def lengths(items):
        return {label:length_summary([v for v in items if outcome is None or v["Y"]==outcome]) for label,outcome in (("all",None),("Y0",0),("Y1",1))}
    return dict(root_rows=root_rows,task_rows=task_rows,main=main,variability=variability,concentration=concentration,
        decomposition=dict(new_pass_count=sum(r["new_pass_contribution"] for r in root_rows),
            retention_count=sum(r["retention_contribution"] for r in root_rows),
            new_pass_pp=100*sum(r["new_pass_contribution"] for r in root_rows)/(nr*n),
            retention_pp=100*sum(r["retention_contribution"] for r in root_rows)/(nr*n)),
        sensitivity=loo,categories=categories,transitions=transitions,
        lengths=dict(states={s:lengths([table[s,t] for t in task_ids]) for s in states},
            arms={a:lengths([v for (s,t),v in table.items() if (s=="BASE" if a=="BASE" else s.endswith("_"+a))]) for a in ("BASE","POL_KEEP","POL_FILTER")},
            quantiles="type7_linear",finish_reason="unknown; 1024 does not establish truncation"))


def summarize_nat_replace(rows, *, roots, task_ids):
    """Eight paired training roots on fixed E285; BASE is descriptive only."""
    if len(roots) != 8 or len(set(roots)) != 8 or len(task_ids) != 285 or len(set(task_ids)) != 285:
        raise ValueError("NAT-REPLACE requires eight distinct roots and E285")
    states = ["BASE"] + [f"{r}_{a}" for r in roots for a in ("POL_KEEP", "POL_FILTER")]
    values = {(s, t): None for s in states for t in task_ids}
    seen = set()
    for row in rows:
        key, y = (row["state_id"], row["task_id"]), row["Y"]
        if key in seen:
            raise ValueError("NAT-REPLACE duplicate position")
        if key not in values or (y is not None and (type(y) is not int or y not in (0, 1))):
            raise ValueError("NAT-REPLACE unknown position or nonbinary outcome")
        seen.add(key)
        values[key] = y

    def bounds(terms, denominator):
        low = high = 0
        for key, sign in terms:
            y = values[key]
            low += sign*y if y is not None else min(0, sign)
            high += sign*y if y is not None else max(0, sign)
        return [low/denominator, high/denominator]

    def arm(state):
        ys = [values[state, t] for t in task_ids]
        passed, valid = sum(y == 1 for y in ys), sum(y is not None for y in ys)
        return dict(passed=passed, valid=valid, missing=285-valid,
                    pass_rate=passed/285 if valid == 285 else None,
                    pass_rate_bounds=[passed/285, (passed+285-valid)/285])

    root_rows = []
    for root in roots:
        k, f = f"{root}_POL_KEEP", f"{root}_POL_FILTER"
        ka, fa = arm(k), arm(f)
        cells = dict(both_pass=0, K_only=0, F_only=0, both_fail=0, missing_pairs=0)
        for t in task_ids:
            ky, fy = values[k, t], values[f, t]
            cell = ("missing_pairs" if ky is None or fy is None else "both_pass" if ky and fy
                    else "K_only" if ky else "F_only" if fy else "both_fail")
            cells[cell] += 1
        db = bounds([((s, t), sign) for t in task_ids for s, sign in ((f, 1), (k, -1))], 285)
        root_rows.append(dict(root=root, K_passed=ka["passed"], F_passed=fa["passed"],
            K_valid=ka["valid"], F_valid=fa["valid"], K_pass_rate=ka["pass_rate"], F_pass_rate=fa["pass_rate"],
            **cells, delta=db[0] if cells["missing_pairs"] == 0 else None,
            delta_lower=db[0], delta_upper=db[1]))
    contributions = []
    for t in task_ids:
        db = bounds([((f"{r}_{a}", t), sign) for r in roots
                     for a, sign in (("POL_FILTER", 1), ("POL_KEEP", -1))], 1)
        complete = all(values[f"{r}_{a}", t] is not None for r in roots for a in ("POL_KEEP", "POL_FILTER"))
        contributions.append(dict(task_id=t, net=int(db[0]) if complete else None,
            net_lower=int(db[0]), net_upper=int(db[1]), delta_contribution=db[0]/2280 if complete else None))
    valid = sum(y is not None for y in values.values())
    primary_complete = all(r["missing_pairs"] == 0 for r in root_rows)
    db = [sum(r[f"delta_{side}"] for r in root_rows)/8 for side in ("lower", "upper")]
    result = dict(complete=valid == 4845, primary_complete=primary_complete, valid_positions=valid,
        total_positions=4845, primary_valid_positions=valid-arm("BASE")["valid"],
        delta=None, delta_pp=None, delta_bounds=db, ci95=None, ci95_pp=None, sample_sd=None,
        scientific_disposition="PRIMARY_INCOMPLETE", root_rows=root_rows,
        task_contributions=contributions, BASE=arm("BASE"), concentration=None,
        inference_unit="paired_training_root", n_roots=8, fixed_endpoint_denominator=285, df=7)
    if not primary_complete:
        return result
    ds = [r["delta"] for r in root_rows]
    mean = sum(ds)/8
    sd = math.sqrt(sum((d-mean)**2 for d in ds)/7)
    interval = [mean-2.364624251*sd/math.sqrt(8), mean+2.364624251*sd/math.sqrt(8)] if sd > 0 else None
    disposition = "UNRESOLVED_WITHIN_BUDGET"
    if interval:
        lo, hi = interval
        disposition = ("ADVERSE_EXPLORATORY" if hi < 0 else
            "POSITIVE_EXPLORATORY" if lo > 0 and mean >= .02 else
            "BELOW_INVESTMENT_SCALE" if hi < .02 else "UNRESOLVED_WITHIN_BUDGET")
    magnitudes = sorted((abs(t["net"]) for t in contributions), reverse=True)
    total_abs = sum(magnitudes)
    result.update(delta=mean, delta_pp=100*mean, sample_sd=sd, ci95=interval,
        ci95_pp=[100*x for x in interval] if interval else None, scientific_disposition=disposition,
        concentration=dict(positive_tasks=sum(t["net"] > 0 for t in contributions),
            negative_tasks=sum(t["net"] < 0 for t in contributions),
            zero_tasks=sum(t["net"] == 0 for t in contributions),
            positive_roots=sum(d > 0 for d in ds), negative_roots=sum(d < 0 for d in ds),
            absolute_net_sum=total_abs,
            top1_absolute_share=magnitudes[0]/total_abs if total_abs else None,
            top5_absolute_share=sum(magnitudes[:5])/total_abs if total_abs else None))
    return result


def summarize_branch_pc(rows, *, pair_ids):
    """Fixed finite-panel arithmetic; unknown binary outcomes retain signed bounds."""
    roots = [312496836, 874163325, 1214513118, 1930090148]
    states = ["BASE"] + [f"{r}_{a}" for r in roots for a in ("R", "M")]
    conditions = ("U", "CR", "CM")
    if len(pair_ids) != 39 or len(set(pair_ids)) != 39:
        raise ValueError("BRANCH requires fixed D39")
    values = {(s, p, c): None for s in states for p in pair_ids for c in conditions}
    seen = set()
    for row in rows:
        key = (row["state_id"], row["pair_id"], row["condition"])
        if key in seen:
            raise ValueError("BRANCH duplicate logical position")
        if key not in values or row["Y"] not in (None, 0, 1):
            raise ValueError("BRANCH unknown position or outcome")
        seen.add(key)
        values[key] = row["Y"]
    valid = sum(y is not None for y in values.values())

    def contrast(terms, denominator):
        low = high = 0
        for key, coefficient in terms:
            y = values[key]
            low += coefficient*y if y is not None else min(0, coefficient)
            high += coefficient*y if y is not None else max(0, coefficient)
        return [low/denominator, high/denominator]

    u_terms = [((f"{r}_{a}", p, "U"), 1 if a == "R" else -1)
               for r in roots for p in pair_ids for a in ("R", "M")]
    b_terms = [((f"{r}_{a}", p, c), 1 if c == "CR" else -1)
               for r in roots for p in pair_ids for a in ("R", "M") for c in ("CR", "CM")]
    base_terms = [(("BASE", p, c), 1 if c == "CR" else -1) for p in pair_ids for c in ("CR", "CM")]
    result = {"complete": valid == 1053, "valid_positions": valid, "total_positions": 1053,
              "completion_fraction": valid/1053, "scientific_disposition": "not_evaluated",
              "Delta_U_bounds": contrast(u_terms, 156), "G_bounds": contrast(b_terms, 312),
              "G_BASE_bounds": contrast(base_terms, 39)}
    result["pass_rates"] = {s: {c: {"passed": sum(values[s,p,c] == 1 for p in pair_ids),
        "valid": sum(values[s,p,c] is not None for p in pair_ids),
        "bounds": contrast([((s,p,c), 1) for p in pair_ids], 39)} for c in conditions} for s in states}
    if not result["complete"]:
        return result

    def cells(left, right):
        counts = {"both_pass": 0, "left_only": 0, "right_only": 0, "both_fail": 0}
        for l, r in zip(left, right):
            counts["both_pass" if l and r else "left_only" if l else "right_only" if r else "both_fail"] += 1
        counts["net"] = counts["left_only"]-counts["right_only"]
        return counts

    root_rows = []
    for r in roots:
        item = {"root": r, "conditions": {}}
        for c in conditions:
            left, right = [[values[f"{r}_{a}",p,c] for p in pair_ids] for a in ("R", "M")]
            item["conditions"][c] = {**cells(left, right), "R_passed": sum(left), "M_passed": sum(right),
                "delta": (sum(left)-sum(right))/39,
                "R_minus_BASE": (sum(left)-sum(values["BASE",p,c] for p in pair_ids))/39,
                "M_minus_BASE": (sum(right)-sum(values["BASE",p,c] for p in pair_ids))/39}
        item["branch_cells"] = {a: cells([values[f"{r}_{a}",p,"CR"] for p in pair_ids],
                                         [values[f"{r}_{a}",p,"CM"] for p in pair_ids]) for a in ("R", "M")}
        item["G_net"] = sum(v["net"] for v in item["branch_cells"].values())
        item["G"] = item["G_net"]/78
        root_rows.append(item)
    per_pair = [{"pair_id": p,
        "d_net": sum(values[f"{r}_R",p,"U"]-values[f"{r}_M",p,"U"] for r in roots),
        "g_net": sum(values[f"{r}_{a}",p,"CR"]-values[f"{r}_{a}",p,"CM"] for r in roots for a in ("R", "M"))}
        for p in pair_ids]
    for p in per_pair:
        p.update(d=p["d_net"]/4, g=p["g_net"]/8)
    n_u = sum(p["d_net"] for p in per_pair)
    n_b = sum(p["g_net"] for p in per_pair)
    positive_d = sum(p["d_net"] > 0 for p in per_pair)
    negative_d = sum(p["d_net"] < 0 for p in per_pair)
    positive_g = sum(p["g_net"] > 0 for p in per_pair)
    positive_roots = sum(r["conditions"]["U"]["net"] > 0 for r in root_rows)
    negative_roots = sum(r["conditions"]["U"]["net"] < 0 for r in root_rows)
    u_plus = n_u >= 8 and positive_d >= 3 and positive_roots >= 3
    u_minus = n_u <= -8 and negative_d >= 3 and negative_roots >= 3
    b_plus = n_b >= 16 and positive_g >= 3
    disposition = ("OPERATIONAL_FLOOR" if all(y == 0 for y in values.values()) else
        "PROMISING_CONDITIONAL_LINK" if u_plus and b_plus else
        "CONDITIONAL_TRAINING_DIFFERENCE_WITHOUT_BRANCH_SUPPORT" if u_plus else
        "OPPOSITE_DIRECTION" if u_minus else "NO_MEANINGFUL_FINITE_PANEL_LINK" if abs(n_u) < 8 else
        "HETEROGENEOUS_OR_UNRESOLVED")
    result.update(scientific_disposition=disposition, N_U=n_u, N_B=n_b, Delta_U=n_u/156, G=n_b/312,
        G_BASE=result["G_BASE_bounds"][0], U_plus=u_plus, U_minus=u_minus, B_plus=b_plus,
        root_rows=root_rows, per_pair=per_pair,
        coverage={"positive_d": positive_d, "negative_d": negative_d, "positive_g": positive_g,
                  "positive_U_roots": positive_roots, "negative_U_roots": negative_roots,
                  "positive_d_and_g": sum(p["d_net"] > 0 and p["g_net"] > 0 for p in per_pair)},
        investment_conditions={"N_U_ge_8": n_u >= 8, "positive_d_ge_3": positive_d >= 3,
            "positive_U_roots_ge_3": positive_roots >= 3, "N_B_ge_16": n_b >= 16,
            "positive_g_ge_3": positive_g >= 3, "N_U_le_minus8": n_u <= -8,
            "negative_d_ge_3": negative_d >= 3, "negative_U_roots_ge_3": negative_roots >= 3},
        BASE_branch_cells=cells([values["BASE",p,"CR"] for p in pair_ids], [values["BASE",p,"CM"] for p in pair_ids]))
    return result


def summarize_scope_pc_nll(suites: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    """Per-topic normalization, separate masks, and explicit cross-text intersections."""
    result = {}
    for job, records in suites.items():
        if not records or "nll" not in records[0]:
            continue
        result[job] = {}
        for role in ("T64", "D"):
            pairs = [row for row in records if row["role"] == role]
            regions = result[job][role] = {}
            for region in ("all", "prefix", "difference", "suffix"):
                values = {text: {row["pair_id"]: row["nll"][text][region] for row in pairs
                    if region in row["nll"][text] and row["nll"][text][region]["mean"] is not None}
                    for text in ("R", "M")}
                regions[region] = {text: {"mean": float(np.mean([v["mean"] for v in items.values()])) if items else None,
                    "pair_count": len(items), "token_count": sum(v["token_count"] for v in items.values()),
                    "pair_ids": sorted(items)} for text, items in values.items()}
                intersection = sorted(set(values["R"]) & set(values["M"]))
                regions[region]["R-M"] = {"mean": float(np.mean([values["R"][key]["mean"] - values["M"][key]["mean"]
                    for key in intersection])) if intersection else None, "pair_count": len(intersection), "pair_ids": intersection}
    return result


def summarize_scope_pc_learning(suites: Mapping[str, Sequence[Mapping[str, object]]], *,
                                roots: Sequence[int]) -> dict[str, object]:
    """Keep each signed pair contrast and each arm's BASE change; no quality gate."""
    indexed = {job: {row["pair_id"]: row for row in rows} for job, rows in suites.items()}
    result = {}
    for role in ("T64", "D"):
        result[role] = {}
        base = {key: row for key, row in indexed.get("BASE", {}).items() if row["role"] == role}
        for step in (25, 100):
            by_root = result[role][str(step)] = {}
            for root in roots:
                r = {key: row for key, row in indexed.get(f"{root}_R_{step}", {}).items() if row["role"] == role}
                m = {key: row for key, row in indexed.get(f"{root}_M_{step}", {}).items() if row["role"] == role}
                if not base or set(base) != set(r) or set(base) != set(m):
                    by_root[str(root)] = None
                    continue
                values = {key: float(r[key]["m"]) - float(m[key]["m"]) for key in base}
                array = list(values.values())
                by_root[str(root)] = {"s": values, "A": float(np.mean(array)), "denominator": len(array),
                    "R-BASE": {key: float(r[key]["m"]) - float(base[key]["m"]) for key in base},
                    "M-BASE": {key: float(m[key]["m"]) - float(base[key]["m"]) for key in base},
                    "distribution": {"median": float(np.median(array)), "q25": float(np.quantile(array, .25)),
                        "q75": float(np.quantile(array, .75)), "min": min(array), "max": max(array),
                        "positive": sum(v > 0 for v in array), "negative": sum(v < 0 for v in array),
                        "zero": sum(v == 0 for v in array)}}
    return result


def summarize_scope_pc_capability(panels: Mapping[str, Sequence[bool]], *, roots: Sequence[int], denominator: int) -> dict[str, object]:
    """Descriptive four-root paired contrasts; absent panels are never zero-filled."""
    if denominator <= 0 or len(set(roots)) != 4:
        raise ValueError("SCOPE capability denominator/roots invalid")
    for values in panels.values():
        if len(values) != denominator or any(type(value) is not bool for value in values):
            raise ValueError("SCOPE capability requires a complete keyed panel")
    base = sum(panels["BASE"]) / denominator if "BASE" in panels else None
    records = {}
    for root in roots:
        r, m = panels.get(f"{root}/R"), panels.get(f"{root}/M")
        y_r = sum(r) / denominator if r is not None else None
        y_m = sum(m) / denominator if m is not None else None
        cells = None
        if r is not None and m is not None:
            cells = {"both_pass": sum(a and b for a, b in zip(r, m)),
                     "both_fail": sum(not a and not b for a, b in zip(r, m)),
                     "r_only": sum(a and not b for a, b in zip(r, m)),
                     "m_only": sum(not a and b for a, b in zip(r, m))}
        records[str(root)] = {"R": y_r, "M": y_m, "BASE": base, "four_cells": cells,
            "R-M": (cells["r_only"] - cells["m_only"]) / denominator if cells else None,
            "R-BASE": y_r - base if y_r is not None and base is not None else None,
            "BASE-M": base - y_m if y_m is not None and base is not None else None}
    summary = {}
    for contrast in ("R-M", "R-BASE", "BASE-M"):
        values = [row[contrast] for row in records.values()]
        summary[contrast] = None if any(v is None for v in values) else {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": min(values), "max": max(values), "sample_sd": float(np.std(values, ddof=1))}
    return {"roots": records, "BASE": base, "denominator": denominator,
            "complete_root_count": sum(row["four_cells"] is not None for row in records.values()),
            "summary": summary, "inference": "descriptive_only_no_p_or_ci"}


def _decision(value: float) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("CodeContests statistical result is nonfinite")
    with localcontext() as context:
        context.prec = 50
        return Decimal.from_float(value).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)


def _contrast(left: Sequence[Fraction], right: Sequence[Fraction]) -> dict[str, object]:
    differences = tuple(a - b for a, b in zip(left, right, strict=True))
    mean_fraction = sum(differences, Fraction()) / 8
    values = np.asarray([float(value) for value in differences], dtype=np.float64)
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(8.0)
    from scipy import stats

    half_width = float(stats.t.ppf(0.975, 7)) * standard_error
    mean_float = float(mean_fraction)
    lower = _decision(mean_float - half_width)
    upper = _decision(mean_float + half_width)
    if mean_fraction >= Fraction(1, 50) and lower > 0:
        classification = "supported_improvement"
    elif mean_fraction <= -Fraction(1, 50) and upper < 0:
        classification = "supported_harm"
    elif lower >= -_MARGIN and upper <= _MARGIN:
        classification = "practical_equivalence"
    else:
        classification = "inconclusive"
    return {
        "differences": [float(value) for value in differences],
        "mean": mean_float,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "interval_lower": float(lower),
        "interval_upper": float(upper),
        "degrees_of_freedom": 7,
        "confidence_level": 0.95,
        "practical_margin": 0.02,
        "classification": classification,
    }


def analyze_codecontests_roots(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Consume exactly 24 complete root-arm aggregate scores."""

    if len(records) != 24:
        raise ValueError("CodeContests analysis requires 24 complete root-arm scores")
    expected = tuple(
        (root, arm) for root in CODECONTESTS_FORMAL_ROOTS for arm in ("I", "R", "N")
    )
    by_job: dict[tuple[int, str], Fraction] = {}
    denominator: int | None = None
    for record in records:
        try:
            root = int(record["root"])
            arm = str(record["arm"])
            passed = record["pass_count"]
            eligible = record["eligible_count"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("CodeContests root score is malformed") from exc
        job = (root, arm)
        if (
            job not in expected
            or job in by_job
            or isinstance(passed, bool)
            or not isinstance(passed, int)
            or isinstance(eligible, bool)
            or not isinstance(eligible, int)
            or eligible < 250
            or not 0 <= passed <= eligible
        ):
            raise ValueError("CodeContests root score identity or count is invalid")
        if denominator is None:
            denominator = eligible
        elif eligible != denominator:
            raise ValueError("CodeContests endpoint denominator differs across roots/arms")
        by_job[job] = Fraction(passed, eligible)
    if tuple(by_job) != expected:
        raise ValueError("CodeContests analysis requires 24 complete ordered root-arm scores")

    values = {
        arm: tuple(by_job[(root, arm)] for root in CODECONTESTS_FORMAL_ROOTS)
        for arm in ("I", "R", "N")
    }
    contrasts = {
        "R-I": _contrast(values["R"], values["I"]),
        "N-I": _contrast(values["N"], values["I"]),
        "R-N": _contrast(values["R"], values["N"]),
    }
    primary = contrasts["R-I"]["classification"] == "supported_improvement"
    corroborating = contrasts["N-I"]["classification"] == "supported_improvement"
    return {
        "schema": "codecontests-v1-j1-analysis-v1",
        "complete": True,
        "root_count": 8,
        "endpoint_denominator": denominator,
        "contrasts": contrasts,
        "claims": {
            "same_problem_repair_improved": primary,
            "incorrect_underperformed_two_correct_alternatives": primary
            and corroborating,
        },
        "diagnostic_rn_can_rescue_claim": False,
        "prompt_level_tests_performed": False,
    }


_V2_PRIMARY_ARMS = {
    "positive-control": ("PC_CLEAN", "PC_MUTANT"),
    "natural-certificate": ("NAT_R", "NAT_I"),
    "policy-filter": ("POL_FILTER", "POL_KEEP"),
}


def _v2_contrast(left: Sequence[Fraction], right: Sequence[Fraction]) -> dict[str, object]:
    differences = tuple(a - b for a, b in zip(left, right, strict=True))
    values = np.asarray([float(value) for value in differences], dtype=np.float64)
    mean_fraction = sum(differences, Fraction()) / 12
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(12.0)
    from scipy import stats

    half_width = float(stats.t.ppf(0.975, 11)) * standard_error
    lower = _decision(float(mean_fraction) - half_width)
    upper = _decision(float(mean_fraction) + half_width)
    if lower > 0:
        directional = "improvement_supported"
    elif upper < 0:
        directional = "harm_supported"
    else:
        directional = "direction_inconclusive"
    if lower >= _MARGIN:
        practical = "practical_improvement_supported"
    elif upper <= -_MARGIN:
        practical = "practical_harm_supported"
    elif lower > -_MARGIN and upper < _MARGIN:
        practical = "practical_equivalence_supported"
    else:
        practical = "practical_unresolved"
    return {
        "differences": [float(value) for value in differences],
        "mean": float(mean_fraction),
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "interval_lower": float(lower),
        "interval_upper": float(upper),
        "degrees_of_freedom": 11,
        "confidence_level": 0.95,
        "practical_margin": 0.02,
        "directional_status": directional,
        "practical_status": practical,
    }


def analyze_codecontests_v2_stage(
    records: Sequence[Mapping[str, object]], *, stage: str
) -> dict[str, object]:
    """Analyze one reached V2 stage without pooling V1 or downstream outcomes."""

    if stage not in _V2_PRIMARY_ARMS:
        raise ValueError("unknown CodeContests V2 stage")
    roots = CODECONTESTS_V2_FORMAL_ROOTS[stage]
    arms = CODECONTESTS_V2_STAGE_ARMS[stage]
    expected = {(root, arm) for root in roots for arm in arms}
    expected_count = len(expected)
    if len(records) != expected_count:
        raise ValueError(f"CodeContests V2 analysis requires {expected_count} complete root-arm scores")
    by_job: dict[tuple[int, str], Fraction] = {}
    denominator: int | None = None
    for record in records:
        if record.get("protocol_id") != CODECONTESTS_V2_PROTOCOL_ID:
            raise ValueError("CodeContests V2 protocol identity mismatch")
        if record.get("stage") != stage:
            raise ValueError("CodeContests V2 stage identity mismatch")
        try:
            root = int(record["root"])
            arm = str(record["arm"])
            passed = record["pass_count"]
            eligible = record["eligible_count"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("CodeContests V2 root score is malformed") from exc
        job = (root, arm)
        if (
            job not in expected
            or job in by_job
            or isinstance(passed, bool)
            or not isinstance(passed, int)
            or isinstance(eligible, bool)
            or not isinstance(eligible, int)
            or eligible < 250
            or not 0 <= passed <= eligible
        ):
            raise ValueError("CodeContests V2 root score identity or count is invalid")
        if denominator is None:
            denominator = eligible
        elif eligible != denominator:
            raise ValueError("CodeContests V2 endpoint denominator differs")
        by_job[job] = Fraction(passed, eligible)
    if set(by_job) != expected:
        raise ValueError(f"CodeContests V2 analysis requires {expected_count} complete root-arm scores")

    left_arm, right_arm = _V2_PRIMARY_ARMS[stage]
    primary = _v2_contrast(
        tuple(by_job[(root, left_arm)] for root in roots),
        tuple(by_job[(root, right_arm)] for root in roots),
    )
    diagnostics: dict[str, object] = {}
    if stage == "policy-filter":
        diagnostics["POL_REPAIR-POL_KEEP"] = _v2_contrast(
            tuple(by_job[(root, "POL_REPAIR")] for root in roots),
            tuple(by_job[(root, "POL_KEEP")] for root in roots),
        )
        diagnostics["POL_FILTER-POL_REPAIR"] = _v2_contrast(
            tuple(by_job[(root, "POL_FILTER")] for root in roots),
            tuple(by_job[(root, "POL_REPAIR")] for root in roots),
        )
    return {
        "schema": "codecontests-v2-stage-analysis-v1",
        "protocol_id": CODECONTESTS_V2_PROTOCOL_ID,
        "stage": stage,
        "complete": True,
        "root_count": 12,
        "endpoint_denominator": denominator,
        "primary_contrast": f"{left_arm}-{right_arm}",
        "contrast": primary,
        "diagnostic_contrasts": diagnostics,
        "advance_authorized": primary["directional_status"] == "improvement_supported",
        "diagnostics_can_authorize": False,
        "prompt_level_tests_performed": False,
        "v1_results_pooled": False,
    }


def select_s12_confirmatory_root_count(
    anonymous_scores: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Expose only the prespecified four-root variance decision."""

    if len(anonymous_scores) != 4:
        raise ValueError("S12 planning requires exactly four anonymous paired scores")
    differences: list[float] = []
    for score in anonymous_scores:
        if set(score) != {"A", "B"}:
            raise ValueError("S12 planning interface accepts only anonymous A/B scores")
        left, right = score["A"], score["B"]
        if (
            isinstance(left, bool)
            or not isinstance(left, (int, float))
            or isinstance(right, bool)
            or not isinstance(right, (int, float))
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
        ):
            raise ValueError("S12 planning score is nonfinite or malformed")
        differences.append(float(left) - float(right))
    sample_sd = float(np.std(np.asarray(differences, dtype=np.float64), ddof=1))
    if not math.isfinite(sample_sd):
        raise ValueError("S12 planning SD is nonfinite")
    selected_n = 12 if sample_sd <= 0.019673586 else 16
    from scipy import stats

    half_width = float(stats.t.ppf(0.975, selected_n - 1)) * sample_sd / math.sqrt(selected_n)
    return {
        "schema": "codecontests-s12-planning-decision-v1",
        "s_plan": sample_sd,
        "selected_n": selected_n,
        "precision_target_met": half_width <= 0.0125,
    }


_S12_STAGE_ARMS = {
    "PC": ("PC_CLEAN", "PC_MUTANT"),
    "NAT": ("NAT_R", "NAT_I"),
    "POLICY": ("POL_KEEP", "POL_FILTER", "POL_REPAIR"),
}
_S12_CONTRASTS = {
    "PC": (("PC_CLEAN-PC_MUTANT", "PC_CLEAN", "PC_MUTANT"),),
    "NAT": (("NAT_R-NAT_I", "NAT_R", "NAT_I"),),
    "POLICY": (
        ("POL_FILTER-POL_KEEP", "POL_FILTER", "POL_KEEP"),
        ("POL_REPAIR-POL_KEEP", "POL_REPAIR", "POL_KEEP"),
        ("POL_REPAIR-POL_FILTER", "POL_REPAIR", "POL_FILTER"),
    ),
}


def _s12_contrast(left: Sequence[Fraction], right: Sequence[Fraction]) -> dict[str, object]:
    count = len(left)
    differences = tuple(a - b for a, b in zip(left, right, strict=True))
    values = np.asarray([float(value) for value in differences], dtype=np.float64)
    mean_fraction = sum(differences, Fraction()) / count
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(count)
    from scipy import stats

    half_width = float(stats.t.ppf(0.975, count - 1)) * standard_error
    lower = _decision(float(mean_fraction) - half_width)
    upper = _decision(float(mean_fraction) + half_width)
    if lower > 0:
        directional = "improvement_supported"
    elif upper < 0:
        directional = "harm_supported"
    else:
        directional = "direction_inconclusive"
    if lower >= _MARGIN:
        practical = "practical_improvement_supported"
    elif upper <= -_MARGIN:
        practical = "practical_harm_supported"
    elif lower > -_MARGIN and upper < _MARGIN:
        practical = "practical_equivalence_supported"
    else:
        practical = "practical_unresolved"
    leave_one_out = [
        float((sum(differences, Fraction()) - differences[index]) / (count - 1))
        for index in range(count)
    ]
    return {
        "mean": float(mean_fraction),
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "interval_lower": float(lower),
        "interval_upper": float(upper),
        "degrees_of_freedom": count - 1,
        "confidence_level": 0.95,
        "practical_margin": 0.02,
        "directional_status": directional,
        "practical_status": practical,
        "positive_difference_count": sum(value > 0 for value in differences),
        "leave_one_root_out_mean_range": [min(leave_one_out), max(leave_one_out)],
    }


def analyze_s12_confirmatory_stage(
    records: Sequence[Mapping[str, object]],
    *,
    stage: str,
    roots: Sequence[int],
) -> dict[str, object]:
    """Analyze exactly 12 or 16 fresh complete confirmatory paired roots."""

    if stage not in _S12_STAGE_ARMS:
        raise ValueError("unknown S12 stage")
    root_tuple = tuple(int(root) for root in roots)
    if len(root_tuple) not in {12, 16}:
        raise ValueError("S12 confirmatory analysis requires 12 or 16 roots")
    if len(set(root_tuple)) != len(root_tuple):
        raise ValueError("S12 confirmatory roots must be unique")
    arms = _S12_STAGE_ARMS[stage]
    expected = tuple((root, arm) for root in root_tuple for arm in arms)
    if len(records) != len(expected):
        raise ValueError("S12 confirmatory analysis requires complete roots")
    by_job: dict[tuple[int, str], Fraction] = {}
    observed_order: list[tuple[int, str]] = []
    for record in records:
        if (
            record.get("schema") != "codecontests-s12-root-score-v1"
            or record.get("protocol_id") != "codecontests-s12-apcf-v1"
            or record.get("stage") != stage
            or record.get("role") != "confirmatory"
            or record.get("status") != "complete"
        ):
            raise ValueError("S12 confirmatory score identity is invalid")
        root = record.get("root")
        arm = record.get("scientific_arm")
        passed = record.get("pass_count")
        eligible = record.get("eligible_count")
        job = (root, arm)
        if (
            job not in expected
            or job in by_job
            or isinstance(passed, bool)
            or not isinstance(passed, int)
            or eligible != 285
            or not 0 <= passed <= 285
        ):
            raise ValueError("S12 confirmatory root/arm/count is invalid")
        observed_order.append((int(root), str(arm)))
        by_job[(int(root), str(arm))] = Fraction(passed, 285)
    if tuple(observed_order) != expected:
        raise ValueError("S12 confirmatory scores must use frozen root-major arm order")
    values = {
        arm: tuple(by_job[(root, arm)] for root in root_tuple) for arm in arms
    }
    contrasts = {
        name: _s12_contrast(values[left], values[right])
        for name, left, right in _S12_CONTRASTS[stage]
    }
    primary = _S12_CONTRASTS[stage][0][0]
    return {
        "schema": "codecontests-s12-confirmatory-analysis-v1",
        "protocol_id": "codecontests-s12-apcf-v1",
        "stage": stage,
        "root_count": len(root_tuple),
        "endpoint_denominator": 285,
        "primary_contrast": primary,
        "contrasts": contrasts,
        "advance_authorized": contrasts[primary]["directional_status"]
        == "improvement_supported",
        "planning_roots_included": False,
    }


def project_nat_replica_wall_hours(
    *,
    generation_probe_workers: Sequence[Mapping[str, object]],
    generation_concurrency: int,
    generation_history_hours: float | None,
    canonical_c4_hours: float,
    stress_program_seconds: Sequence[float],
    scoring_history_hours: float | None,
) -> dict[str, object]:
    """Apply the pre-registered NAT-REPLICA-X164 wall-clock projection."""

    if generation_concurrency not in (1, 2, 4):
        raise ValueError("generation concurrency must be one of 1, 2 or 4")
    expected_workers = generation_concurrency
    if len(generation_probe_workers) != expected_workers:
        raise ValueError("generation probe worker count differs from concurrency")
    if len(stress_program_seconds) != 12:
        raise ValueError("resource projection requires exactly 12 stress programs")
    if canonical_c4_hours <= 0 or any(float(value) <= 0 for value in stress_program_seconds):
        raise ValueError("resource timing observations must be positive")

    loads: list[float] = []
    decode_per_position: list[float] = []
    for row in generation_probe_workers:
        positions = row.get("positions")
        load = row.get("load_seconds")
        decode = row.get("decode_seconds")
        if (
            isinstance(positions, bool)
            or not isinstance(positions, int)
            or positions <= 0
            or not isinstance(load, (int, float))
            or not isinstance(decode, (int, float))
            or float(load) < 0
            or float(decode) <= 0
        ):
            raise ValueError("generation probe telemetry is invalid")
        loads.append(float(load))
        decode_per_position.append(float(decode) / positions)

    target_positions = 17 * 164
    historical_positions = 17 * 285
    generation_probe_typical = (
        math.ceil(17 / generation_concurrency)
        * (max(loads) + 164 * max(decode_per_position))
        / 3600
    )
    generation_history_typical = (
        None
        if generation_history_hours is None
        else float(generation_history_hours) * target_positions / historical_positions
    )
    sorted_stress = sorted(float(value) for value in stress_program_seconds)
    stress_p95_seconds = sorted_stress[math.ceil(0.95 * len(sorted_stress)) - 1]
    score_canonical_typical = 17 * float(canonical_c4_hours)
    score_stress_typical = (
        math.ceil(target_positions / 4) * stress_p95_seconds / 3600
    )
    score_history_typical = (
        None
        if scoring_history_hours is None
        else float(scoring_history_hours) * target_positions / historical_positions
    )

    generation_probe_upper = 1.5 * generation_probe_typical
    generation_history_upper = (
        None if generation_history_typical is None else 1.25 * generation_history_typical
    )
    score_canonical_upper = 1.5 * score_canonical_typical
    score_stress_upper = 1.5 * score_stress_typical
    score_history_upper = (
        None if score_history_typical is None else 1.25 * score_history_typical
    )
    train_typical = 2.107
    train_upper = 1.25 * train_typical
    fixed = 0.75
    generation_typical = max(
        value for value in (generation_probe_typical, generation_history_typical) if value is not None
    )
    generation_upper = max(
        value for value in (generation_probe_upper, generation_history_upper) if value is not None
    )
    score_typical = max(
        value
        for value in (score_canonical_typical, score_stress_typical, score_history_typical)
        if value is not None
    )
    score_upper = max(
        value
        for value in (score_canonical_upper, score_stress_upper, score_history_upper)
        if value is not None
    )
    history_complete = generation_history_hours is not None and scoring_history_hours is not None
    formal_upper = train_upper + generation_upper + score_upper + fixed
    return {
        "schema": "nat-replica-x164-resource-projection-v1",
        "inputs": {
            "target_positions": target_positions,
            "historical_positions": historical_positions,
            "generation_concurrency": generation_concurrency,
            "validation_concurrency": 4,
            "stress_fixture_count": 12,
            "stress_p95_seconds": stress_p95_seconds,
            "history_complete": history_complete,
        },
        "typical": {
            "train_hours": train_typical,
            "generation_probe_hours": generation_probe_typical,
            "generation_history_hours": generation_history_typical,
            "generation_hours": generation_typical,
            "score_canonical_hours": score_canonical_typical,
            "score_stress_hours": score_stress_typical,
            "score_history_hours": score_history_typical,
            "score_hours": score_typical,
            "fixed_hours": fixed,
            "formal_hours": train_typical + generation_typical + score_typical + fixed,
        },
        "upper": {
            "train_hours": train_upper,
            "generation_probe_hours": generation_probe_upper,
            "generation_history_hours": generation_history_upper,
            "generation_hours": generation_upper,
            "score_canonical_hours": score_canonical_upper,
            "score_stress_hours": score_stress_upper,
            "score_history_hours": score_history_upper,
            "score_hours": score_upper,
            "fixed_hours": fixed,
            "formal_hours": formal_upper,
        },
        "resource_pass": history_complete and formal_upper <= 8.0,
    }


def _nat_replica_exact_hash(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("exact overlap inputs must be strings")
    lf = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in lf.split("\n")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def analyze_nat_replica_panel_overlap(
    humaneval_records: Sequence[Mapping[str, object]],
    training_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare the fixed HumanEval panel and S230 I/N units by exact identities only."""

    expected_ids = tuple(f"HumanEval/{index}" for index in range(164))
    human_ids = tuple(row.get("task_id") for row in humaneval_records)
    if human_ids != expected_ids:
        raise ValueError("panel overlap requires ordered HumanEval/0..163")
    if len(training_records) != 460:
        raise ValueError("panel overlap requires exactly 460 I/N training units")
    pair_roles: dict[str, set[str]] = {}
    for row in training_records:
        pair_id = row.get("pair_id")
        role = row.get("role")
        if not isinstance(pair_id, str) or role not in {"I", "N"}:
            raise ValueError("training overlap identity is invalid")
        pair_roles.setdefault(pair_id, set()).add(str(role))
        for field in ("problem_id", "prompt", "program"):
            if not isinstance(row.get(field), str):
                raise ValueError("training overlap text is missing")
    if len(pair_roles) != 230 or any(roles != {"I", "N"} for roles in pair_roles.values()):
        raise ValueError("training overlap requires 230 paired I/N identities")

    human_prompt_hashes = {_nat_replica_exact_hash(str(row["prompt"])) for row in humaneval_records}
    human_program_hashes = {
        _nat_replica_exact_hash(str(row["prompt"]) + str(row["canonical_solution"]))
        for row in humaneval_records
    }
    human_suffix_hashes = {
        _nat_replica_exact_hash(str(row["canonical_solution"])) for row in humaneval_records
    }
    training_prompt_hashes = {
        _nat_replica_exact_hash(str(row["prompt"])) for row in training_records
    }
    training_program_hashes = {
        _nat_replica_exact_hash(str(row["program"])) for row in training_records
    }
    training_ids = {str(row["problem_id"]) for row in training_records}
    counts = {
        "task_id": len(set(expected_ids) & training_ids),
        "prompt": len(human_prompt_hashes & training_prompt_hashes),
        "program": len(human_program_hashes & training_program_hashes),
    }
    return {
        "schema": "nat-replica-x164-exact-overlap-v1",
        "normalization": "CRLF_to_LF_then_rstrip_each_line_then_strip_UTF8_SHA256",
        "population": {"humaneval": 164, "training_pairs": 230, "training_units": 460},
        "overlap_counts": counts,
        "canonical_suffix_overlap_count": len(human_suffix_hashes & training_program_hashes),
        "passed": all(value == 0 for value in counts.values()),
        "semantic_or_pretraining_independence_inferred": False,
    }


def validate_nat_replica_strategy(
    *, design: Mapping[str, object], rebuilt: Mapping[str, object]
) -> dict[str, object]:
    """Check that a CPU-only rebuild is the exact frozen NAT K/F strategy."""

    mismatches: list[str] = []
    roots = design.get("roots")
    if not isinstance(roots, list) or len(roots) != 8 or rebuilt.get("roots") != roots:
        mismatches.append("root_identity")
    bindings = rebuilt.get("source_bindings")
    if bindings != design.get("source_bindings"):
        mismatches.append("source_bindings")
    if rebuilt.get("token_totals") != design.get("token_totals") or design.get("token_totals") != {
        "I": 66297,
        "N": 67198,
    }:
        mismatches.append("token_totals")

    plans = rebuilt.get("plans")
    expected_plans = design.get("plans")
    expected_order = (
        [(root, arm) for root in roots for arm in ("POL_KEEP", "POL_FILTER")]
        if isinstance(roots, list)
        else []
    )
    observed_order = (
        [(row.get("root"), row.get("arm")) for row in plans]
        if isinstance(plans, list) and all(isinstance(row, dict) for row in plans)
        else []
    )
    if plans != expected_plans or observed_order != expected_order:
        mismatches.append("plan_order_and_identity")
    training = rebuilt.get("training_recipe")
    fixed_training = {
        "base_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "base_dtype": "BF16",
        "lora_dtype": "FP32",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "optimizer": "AdamW",
        "max_length": 2048,
        "microbatch": 1,
        "updates": 100,
        "B80_tokens": 467180,
        "component20_tokens": 116795,
        "response_tokens": 583975,
        "tf32": False,
    }
    if training != design.get("training_recipe") or not isinstance(training, dict) or any(
        training.get(key) != value for key, value in fixed_training.items()
    ):
        mismatches.append("training_recipe")
    generation = rebuilt.get("generation_recipe")
    fixed_generation = {
        "add_special_tokens": True,
        "do_sample": False,
        "max_new_tokens": 1024,
        "microbatch": 1,
        "eos_token_id": 151643,
        "pad_token_id": 151643,
        "suffix_only": True,
        "tabs_to_four_spaces": True,
    }
    if generation != design.get("generation_recipe") or not isinstance(generation, dict) or any(
        generation.get(key) != value for key, value in fixed_generation.items()
    ):
        mismatches.append("generation_recipe")

    role_counts = {role: 0 for role in ("I", "N")}
    pair_ids: set[str] = set()
    if isinstance(bindings, list):
        for row in bindings:
            if not isinstance(row, dict) or row.get("role") not in role_counts:
                mismatches.append("source_binding_schema")
                break
            role_counts[str(row["role"])] += 1
            if isinstance(row.get("pair_id"), str):
                pair_ids.add(str(row["pair_id"]))
    if role_counts != {"I": 230, "N": 230} or len(pair_ids) != 230:
        mismatches.append("source_binding_population")
    all_100 = isinstance(plans, list) and len(plans) == 16 and all(
        isinstance(row, dict)
        and row.get("updates") == 100
        and row.get("response_tokens") == 583975
        for row in plans
    )
    if not all_100:
        mismatches.append("plan_dose")
    return {
        "schema": "nat-replica-x164-strategy-check-v1",
        "status": "passed" if not mismatches else "failed",
        "counts": {
            "pairs": len(pair_ids),
            "i_units": role_counts["I"],
            "n_units": role_counts["N"],
            "plans": len(plans) if isinstance(plans, list) else 0,
        },
        "all_100_update_plans": all_100,
        "mismatches": tuple(dict.fromkeys(mismatches)),
        "training_or_generation_performed": False,
    }


def resolve_w259_terminal(
    *,
    complete: bool,
    all_zero_or_full: bool = False,
    mean: float | None = None,
    sd: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> str | None:
    """Apply the frozen W259 scientific terminal order."""

    if not complete:
        return None
    if all_zero_or_full:
        return "W_READOUT_DEGENERATE"
    if sd == 0:
        return "W_UNRESOLVED_WITHIN_BUDGET"
    if None in (mean, sd, lower, upper):
        raise ValueError("Q1-W259-PROSPECTIVE complete analysis lacks interval facts")
    if lower > 0 and mean >= 2:
        return "W_DIRECTIONAL_SUPPORT_EXPLORATORY"
    if upper < 0:
        return "W_DIRECTION_REVERSED_EXPLORATORY"
    if upper < 2:
        return "W_BELOW_INVESTMENT_SCALE"
    return "W_UNRESOLVED_WITHIN_BUDGET"


def analyze_w259_positions(
    rows: Sequence[Mapping[str, object]],
    *,
    pairs: Sequence[tuple[str, int, int, str]],
    task_ids: Sequence[str],
) -> dict[str, object]:
    """Compute the finite eight-pair OMIT-minus-USE readout."""

    tasks = tuple(str(task_id) for task_id in task_ids)
    if len(tasks) != 164 or len(set(tasks)) != 164:
        raise ValueError("Q1-W259-PROSPECTIVE endpoint must contain 164 tasks")
    pair_specs = {
        str(pair_id): int(root) for pair_id, _seed, root, _digest in pairs
    }
    if len(pair_specs) != 8:
        raise ValueError("Q1-W259-PROSPECTIVE needs eight unique pairs")
    expected = {
        (pair_id, arm, task_id)
        for pair_id in pair_specs
        for arm in ("W_USE", "W_OMIT")
        for task_id in tasks
    }
    by_key: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row.get("pair_id")), str(row.get("arm")), str(row.get("task_id")))
        if key not in expected or key in by_key:
            raise ValueError("Q1-W259-PROSPECTIVE position identity is foreign or duplicated")
        if row.get("root") != pair_specs[key[0]]:
            raise ValueError("Q1-W259-PROSPECTIVE position root differs from its pair")
        if row.get("Y") not in (0, 1, None):
            raise ValueError("Q1-W259-PROSPECTIVE Y is outside {0,1,missing}")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("Q1-W259-PROSPECTIVE position grid is incomplete")

    root_rows: list[dict[str, object]] = []
    root_d: list[float] = []
    root_net: list[int] = []
    all_observed_y: list[int] = []
    for pair_id, _seed, root, _digest in pairs:
        counts: dict[str, tuple[int, int]] = {}
        for arm in ("W_USE", "W_OMIT"):
            values = [by_key[(pair_id, arm, task_id)].get("Y") for task_id in tasks]
            valid = [int(value) for value in values if value in (0, 1)]
            counts[arm] = (sum(valid), len(valid))
            all_observed_y.extend(valid)
        use_pass, use_valid = counts["W_USE"]
        omit_pass, omit_valid = counts["W_OMIT"]
        net = omit_pass - use_pass
        lower = (omit_pass - (use_pass + 164 - use_valid)) / 164 * 100
        upper = (omit_pass + 164 - omit_valid - use_pass) / 164 * 100
        complete_pair = use_valid == omit_valid == 164
        d_pp = net / 164 * 100 if complete_pair else None
        if complete_pair:
            root_net.append(net)
            root_d.append(float(d_pp))
        root_rows.append(
            {
                "pair_id": pair_id,
                "root": root,
                "W_USE_pass": use_pass,
                "W_USE_valid": use_valid,
                "W_OMIT_pass": omit_pass,
                "W_OMIT_valid": omit_valid,
                "missing": 328 - use_valid - omit_valid,
                "net_pass": net if complete_pair else None,
                "d_pp": d_pp,
                "bound_lower_pp": lower,
                "bound_upper_pp": upper,
            }
        )
    complete = len(root_d) == 8
    mean = sum(root_d) / 8 if complete else None
    sd = (
        math.sqrt(sum((value - mean) ** 2 for value in root_d) / 7)
        if complete and mean is not None
        else None
    )
    interval = None
    if complete and sd is not None and sd > 0:
        half_width = 2.364624251 * sd / math.sqrt(8)
        interval = [mean - half_width, mean + half_width]
    all_zero_or_full = complete and bool(all_observed_y) and (
        all(value == 0 for value in all_observed_y)
        or all(value == 1 for value in all_observed_y)
    )
    terminal = resolve_w259_terminal(
        complete=complete,
        all_zero_or_full=all_zero_or_full,
        mean=mean,
        sd=sd,
        lower=interval[0] if interval is not None else None,
        upper=interval[1] if interval is not None else None,
    )
    leave_one = (
        [(sum(root_d) - value) / 7 for value in root_d] if complete else []
    )
    return {
        "complete": complete,
        "terminal": terminal,
        "root_rows": root_rows,
        "root_net_pass": root_net,
        "root_d_pp": root_d,
        "mean_d_pp": mean,
        "sd_d_pp": sd,
        "ci95_pp": interval,
        "leave_one_pair_mean_range_pp": (
            [min(leave_one), max(leave_one)] if leave_one else None
        ),
        "missingness_bound_pp": [
            sum(float(row["bound_lower_pp"]) for row in root_rows) / 8,
            sum(float(row["bound_upper_pp"]) for row in root_rows) / 8,
        ],
        "all_zero_or_full": all_zero_or_full,
    }


def resolve_w259_cf12_main(
    *,
    complete: bool,
    all_zero_or_full: bool = False,
    mean: float | None = None,
    sd: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, str] | None:
    """Return separate frozen scientific and investment labels for one main contrast."""

    if not complete:
        return None
    if all_zero_or_full:
        return {"scientific": "READOUT_DEGENERATE", "investment": "NO"}
    if sd == 0:
        return {
            "scientific": "UNRESOLVED_DEGENERATE_VARIANCE",
            "investment": "NO",
        }
    if None in (mean, sd, lower, upper):
        raise ValueError("Q1-W259-CF12 complete main contrast lacks interval facts")
    if float(lower) > 0:
        return {
            "scientific": "POSITIVE_EXPLORATORY",
            "investment": (
                "ELIGIBLE_FOR_REVIEW"
                if float(mean) >= 2.0
                else "BELOW_POINT_INVESTMENT_SCALE"
            ),
        }
    if float(upper) < 0:
        return {"scientific": "REVERSED_EXPLORATORY", "investment": "NO"}
    return {
        "scientific": (
            "BELOW_INTERVAL_INVESTMENT_SCALE"
            if float(upper) < 2.0
            else "WITHIN_BUDGET_UNRESOLVED"
        ),
        "investment": "NO",
    }


def analyze_w259_cf12_positions(
    rows: Sequence[Mapping[str, object]],
    *,
    contexts: Sequence[Mapping[str, object]],
    task_ids: Sequence[str],
) -> dict[str, object]:
    """Analyze the fixed 12-context, four-arm CF12 position grid."""

    context_rows = tuple(dict(row) for row in contexts)
    context_ids = tuple(str(row.get("context_id")) for row in context_rows)
    if context_ids != tuple(f"C{index:02d}" for index in range(1, 13)):
        raise ValueError("Q1-W259-CF12 analysis needs ordered C01..C12")
    roots = {str(row.get("context_id")): row.get("root") for row in context_rows}
    if any(type(root) is not int or root <= 0 for root in roots.values()) or len(
        set(roots.values())
    ) != 12:
        raise ValueError("Q1-W259-CF12 analysis context roots differ")
    tasks = tuple(str(task_id) for task_id in task_ids)
    if tasks != tuple(f"HumanEval/{index}" for index in range(164)):
        raise ValueError("Q1-W259-CF12 endpoint must be ordered HumanEval/0..163")
    arms = ("B", "BG", "BH", "BW")
    expected = {
        (context_id, arm, task_id)
        for context_id in context_ids
        for arm in arms
        for task_id in tasks
    }
    by_key: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("context_id")),
            str(row.get("arm")),
            str(row.get("task_id")),
        )
        if key not in expected or key in by_key:
            raise ValueError("Q1-W259-CF12 position identity is foreign or duplicated")
        if row.get("root") != roots[key[0]]:
            raise ValueError("Q1-W259-CF12 position root differs from its context")
        if row.get("Y") not in (0, 1, None):
            raise ValueError("Q1-W259-CF12 Y is outside {0,1,missing}")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("Q1-W259-CF12 position grid is incomplete")

    coefficients = {
        "hG": {"B": 1, "BG": -1},
        "hH": {"B": 1, "BH": -1},
        "cG": {"BH": 1, "BW": -1},
        "cH": {"BG": 1, "BW": -1},
        "hW": {"B": 1, "BW": -1},
        "J": {"BG": 1, "BH": 1, "B": -1, "BW": -1},
    }
    root_rows: list[dict[str, object]] = []
    contrast_values: dict[str, list[float]] = {name: [] for name in coefficients}
    contrast_bounds: dict[str, list[tuple[float, float]]] = {
        name: [] for name in coefficients
    }
    all_complete = True
    main_observed: dict[str, list[int]] = {"hG": [], "hH": []}
    for context_id in context_ids:
        count_by_arm: dict[str, tuple[int, int]] = {}
        arm_bounds: dict[str, tuple[float, float]] = {}
        for arm in arms:
            values = [by_key[(context_id, arm, task_id)].get("Y") for task_id in tasks]
            observed = [int(value) for value in values if value in (0, 1)]
            passed = sum(observed)
            valid = len(observed)
            count_by_arm[arm] = (passed, valid)
            arm_bounds[arm] = (100 * passed / 164, 100 * (passed + 164 - valid) / 164)
        complete_context = all(valid == 164 for _passed, valid in count_by_arm.values())
        all_complete = all_complete and complete_context
        result: dict[str, object] = {
            "context_id": context_id,
            "root": roots[context_id],
            "missing": sum(164 - valid for _passed, valid in count_by_arm.values()),
        }
        for arm in arms:
            passed, valid = count_by_arm[arm]
            result[f"{arm}_pass"] = passed
            result[f"{arm}_valid"] = valid
        for name, terms in coefficients.items():
            lower = sum(
                coefficient
                * (arm_bounds[arm][0] if coefficient > 0 else arm_bounds[arm][1])
                for arm, coefficient in terms.items()
            )
            upper = sum(
                coefficient
                * (arm_bounds[arm][1] if coefficient > 0 else arm_bounds[arm][0])
                for arm, coefficient in terms.items()
            )
            contrast_bounds[name].append((lower, upper))
            value = (
                100
                * sum(
                    coefficient * count_by_arm[arm][0]
                    for arm, coefficient in terms.items()
                )
                / 164
                if complete_context
                else None
            )
            result[f"{name}_pp"] = value
            result[f"{name}_bound_lower_pp"] = lower
            result[f"{name}_bound_upper_pp"] = upper
            if value is not None:
                contrast_values[name].append(value)
        if complete_context:
            if not math.isclose(
                float(result["J_pp"]),
                float(result["hW_pp"]) - float(result["hG_pp"]) - float(result["hH_pp"]),
                abs_tol=1e-12,
            ) or not math.isclose(
                float(result["J_pp"]),
                float(result["cG_pp"]) - float(result["hG_pp"]),
                abs_tol=1e-12,
            ) or not math.isclose(
                float(result["J_pp"]),
                float(result["cH_pp"]) - float(result["hH_pp"]),
                abs_tol=1e-12,
            ):
                raise ValueError("Q1-W259-CF12 contrast identity differs")
            for name, compared_arms in {"hG": ("B", "BG"), "hH": ("B", "BH")}.items():
                for arm in compared_arms:
                    main_observed[name].extend(
                        int(by_key[(context_id, arm, task_id)]["Y"])
                        for task_id in tasks
                    )
        root_rows.append(result)

    t95 = 2.200985160082949
    t_bonferroni = 2.593092682539353
    contrast_results: dict[str, dict[str, object]] = {}
    for name in coefficients:
        values = contrast_values[name]
        complete = all_complete and len(values) == 12
        mean = sum(values) / 12 if complete else None
        sd = (
            math.sqrt(sum((value - mean) ** 2 for value in values) / 11)
            if complete and mean is not None
            else None
        )
        ci95 = (
            [mean - t95 * sd / math.sqrt(12), mean + t95 * sd / math.sqrt(12)]
            if complete and sd is not None and sd > 0
            else None
        )
        bonferroni = (
            [
                mean - t_bonferroni * sd / math.sqrt(12),
                mean + t_bonferroni * sd / math.sqrt(12),
            ]
            if name in {"hG", "hH"} and complete and sd is not None and sd > 0
            else None
        )
        leave_one = (
            [(sum(values) - value) / 11 for value in values] if complete else []
        )
        contrast_results[name] = {
            "n": 12 if complete else len(values),
            "mean_pp": mean,
            "sample_sd_pp": sd,
            "positive_count": sum(value > 0 for value in values),
            "zero_count": sum(value == 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "ci95_pp": ci95,
            "bonferroni_97_5_ci_pp": bonferroni,
            "leave_one_context_mean_range_pp": (
                [min(leave_one), max(leave_one)] if leave_one else None
            ),
            "missingness_bound_pp": [
                sum(bound[0] for bound in contrast_bounds[name]) / 12,
                sum(bound[1] for bound in contrast_bounds[name]) / 12,
            ],
        }

    labels: dict[str, dict[str, str] | None] | None = None
    terminal = None
    if all_complete:
        labels = {}
        for name in ("hG", "hH"):
            facts = contrast_results[name]
            interval = facts["bonferroni_97_5_ci_pp"]
            observed = main_observed[name]
            labels[name] = resolve_w259_cf12_main(
                complete=True,
                all_zero_or_full=bool(observed)
                and (all(value == 0 for value in observed) or all(value == 1 for value in observed)),
                mean=facts["mean_pp"],
                sd=facts["sample_sd_pp"],
                lower=interval[0] if interval is not None else None,
                upper=interval[1] if interval is not None else None,
            )
        terminal = (
            "CF12_SUBGROUP_SUPPORT_EXPLORATORY"
            if any(
                label is not None and label["investment"] == "ELIGIBLE_FOR_REVIEW"
                for label in labels.values()
            )
            else "CF12_COMPLETE_NO_SUBGROUP_INVESTMENT_SUPPORT"
        )
    return {
        "complete": all_complete,
        "terminal": terminal,
        "context_count": 12,
        "position_count": len(rows),
        "t_critical_95": t95,
        "t_critical_bonferroni_97_5": t_bonferroni,
        "root_rows": root_rows,
        "contrasts": contrast_results,
        "main_labels": labels,
    }
