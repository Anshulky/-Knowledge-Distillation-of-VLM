"""Compare decoder rollout across several checkpoints on the same test documents.

  python -m ie643.rollout.compare --out Data/rollout_cmp

Expects ``<out>/<tag>/*.npz`` from ``python -m ie643.rollout run --tag <tag>`` for every model in MODELS
and ``<out>/cases.json`` ({"better": [...], "worse": [...]}; the remaining uids are the reference sample).

Per document the panel shows the fields whose correctness differs between models (cleaned GT,
ie643.eval.gt_clean), falling back to vendor / date / total. Writes:
  report/docs/<uid>.png   rows = models, columns = input | all values | changed fields
  report/cases.md          per-document F1 and the changed fields' values, model by model
  report/field_metrics.csv per (doc, model, field): image share, area50, teacher-map agreement, correct
  report/summary.json      means per model, split by correct / wrong fields
  report/teacher_agreement.png, report/layer_profile.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ie643.eval.gt_clean import clean_gold
from ie643.eval.match import META_FIELDS
from ie643.eval.score import score_record
from ie643.rollout.analyze import (
    COMPARE_PX,
    GRID,
    INK,
    INK2,
    PANEL_FIELDS,
    _style,
    agreement,
    concentration,
    doc_figure,
    field_rows,
    image_map,
    load_doc,
    mass_split,
    resample,
    safe_name,
)

# tag (rollout subdir) -> (label, prediction key in the test manifest)
MODELS: dict[str, tuple[str, str]] = {
    "base": ("3B base", "student_nohint"),
    "e3a": ("E3a SFT", "student_e3_pred"),
    "e4a": ("E4a KD entropy", "student_e4_pred"),
    "e4_wnone": ("E4 KD w=none", "student_e4_wnone_pred"),
    "e4_wvalue": ("E4 KD w=value", "student_e4_wvalue_pred"),
    "teacher": ("32B teacher", "teacher_nohint"),
}
COLORS = {"base": "#2a78d6", "e3a": "#eb6834", "e4a": "#1baf7a", "e4_wnone": "#eda100", "e4_wvalue": "#e87ba4",
          "teacher": "#4a3aa7"}
MAX_PANEL_FIELDS = 4


def field_outcomes(pred: dict | None, gold: dict) -> dict[str, bool]:
    """field -> correct (no FP and no FN) for every field the cleaned GT scores."""
    return {f: not (fp or fn) for f, (_tp, fp, fn) in score_record(pred, gold).items()}


def doc_f1(pred: dict | None, gold: dict) -> float:
    tp = fp = fn = 0
    for a, b, c in score_record(pred, gold).values():
        tp, fp, fn = tp + a, fp + b, fn + c
    d = 2 * tp + fp + fn
    return 100.0 * 2 * tp / d if d else 100.0


def field_map(doc: dict[str, Any], field: str) -> np.ndarray | None:
    """Map for a top-level field; list fields merge the rows of all their sub-keys (line_items.name, .amount, ...)."""
    groups = field_rows(doc)
    rows = sorted({r for g, rs in groups.items() if g == field or g.startswith(field + ".") for r in rs})
    return image_map(doc, rows) if rows else None


def short(value: Any, n: int = 24) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return f"[{len(value)} rows]"
    return str(value).replace("\n", " ")[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout comparison across checkpoints")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/test/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("Data/rollout_cmp"))
    args = parser.parse_args()
    root = args.manifest.parent
    rows = {r["uid"]: r for r in (json.loads(x) for x in args.manifest.open(encoding="utf-8") if x.strip())}
    cases = json.loads((args.out / "cases.json").read_text())
    group = {u: "new>E4a" for u in cases["better"]} | {u: "new<E4a" for u in cases["worse"]}
    tags = [t for t in MODELS if (args.out / t).is_dir()]
    rep = args.out / "report"
    (rep / "docs").mkdir(parents=True, exist_ok=True)

    uids = sorted({p.stem for t in tags for p in (args.out / t).glob("*.npz")})
    by_safe = {safe_name(u): u for u in rows}
    table: list[dict[str, Any]] = []
    profiles: dict[str, list[np.ndarray]] = {t: [] for t in tags}
    md = ["# Rollout comparison: E4a vs normalised-KD runs", "",
          "F1 = per-document micro F1 on cleaned GT. ✓/✗ = field correct vs cleaned GT; '–' = field not scored.", ""]

    for sname in uids:
        uid = by_safe[sname]
        rec = rows[uid]
        gold = clean_gold(rec.get("source_gt"), rec["dataset"])
        docs = {t: load_doc(args.out / t / f"{sname}.npz") for t in tags if (args.out / t / f"{sname}.npz").exists()}
        preds = {t: rec.get(MODELS[t][1]) for t in docs}
        outcomes = {t: field_outcomes(preds[t], gold) for t in docs}
        f1s = {t: doc_f1(preds[t], gold) for t in docs}
        if uid in group and {"e4a", "e4_wnone", "e4_wvalue"} <= f1s.keys():
            # label from the current scorer, not the one used when cases.json was written
            lo, hi = min(f1s["e4_wnone"], f1s["e4_wvalue"]), max(f1s["e4_wnone"], f1s["e4_wvalue"])
            group[uid] = "new>E4a" if lo > f1s["e4a"] else "new<E4a" if hi < f1s["e4a"] else "tie"

        # fields whose correctness differs between models; E4a-vs-new flips first
        scored = sorted({f for o in outcomes.values() for f in o})
        differ = [f for f in scored if len({outcomes[t].get(f) for t in docs}) > 1 and f not in META_FIELDS]
        key = lambda f: (outcomes.get("e4a", {}).get(f) == outcomes.get("e4_wvalue", {}).get(f), f)  # noqa: E731
        panel_fields = sorted(differ, key=key)[:MAX_PANEL_FIELDS] or list(PANEL_FIELDS)

        H, W = int(rec["height"]), int(rec["width"])
        shared = (H // COMPARE_PX, W // COMPARE_PX)
        maps = {t: {f: image_map(d, r) for f, r in field_rows(d).items()} for t, d in docs.items()}
        for t, d in docs.items():
            profiles[t].append(d["layer_img"] - d["layer_img_sink"])
            for f, m in maps[t].items():
                base_group = f.split(".")[0]
                row = {
                    "uid": uid, "dataset": rec["dataset"], "case": group.get(uid, "reference"), "model": t, "field": f,
                    "correct": outcomes[t].get(base_group) if f != "__all__" else None,
                    **mass_split(d, field_rows(d)[f]), **concentration(m),
                }
                if "teacher" in maps and f in maps["teacher"] and t != "teacher":
                    row.update({f"teacher_{k}": v for k, v in agreement(resample(m, shared),
                                                                     resample(maps["teacher"][f], shared)).items()})
                table.append(row)

        panel: dict[str, list] = {}
        for t in docs:
            cells = [("all values", maps[t].get("__all__"), f"F1 {f1s[t]:.0f}")]
            for f in panel_fields:
                ok = outcomes[t].get(f)
                mark = "" if ok is None else (" ✓" if ok else " ✗")
                m = field_map(docs[t], f)
                cells.append((f, m, short((preds[t] or {}).get(f), 20) + mark))
            panel[MODELS[t][0]] = cells
        sinks = {MODELS[t][0]: (np.nonzero(~d["keep"][d["img_pos"]])[0], tuple(int(x) for x in d["grid"]))
                 for t, d in docs.items()}
        title = f"{uid}  [{group.get(uid, 'reference')}]   × = visual sink"
        doc_figure(root / rec["image"], panel, rep / "docs" / f"{sname}.png", title, sinks)

        md += [f"## {uid}  ({rec['dataset']}, {group.get(uid, 'reference')})", "",
               f"![](docs/{sname}.png)", "",
               "| field | GT | " + " | ".join(MODELS[t][0] for t in docs) + " |",
               "|---|---|" + "---|" * len(docs),
               "| **doc F1** | | " + " | ".join(f"**{f1s[t]:.0f}**" for t in docs) + " |"]
        for f in differ:
            cells = []
            for t in docs:
                ok = outcomes[t].get(f)
                cells.append(f"{short((preds[t] or {}).get(f))} {'✓' if ok else '✗' if ok is not None else '–'}")
            md.append(f"| {f} | {short(gold.get(f))} | " + " | ".join(cells) + " |")
        md.append("")

    with (rep / "field_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in table for k in r}, key=lambda k: (k not in ("uid", "dataset", "case", "model", "field"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)

    def mean(sel: list[dict], k: str) -> float | None:
        vals = [r[k] for r in sel if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    summary: dict[str, Any] = {}
    for t in tags:
        allv = [r for r in table if r["model"] == t and r["field"] == "__all__"]
        fields = [r for r in table if r["model"] == t and r["field"] != "__all__" and r["correct"] is not None]
        summary[t] = {
            "label": MODELS[t][0],
            "all_values": {k: mean(allv, k) for k in ("image", "area50", "entropy_norm", "teacher_spearman", "teacher_top20_iou")},
            "correct_fields": {k: mean([r for r in fields if r["correct"]], k) for k in ("image", "area50", "teacher_top20_iou")},
            "wrong_fields": {k: mean([r for r in fields if not r["correct"]], k) for k in ("image", "area50", "teacher_top20_iou")},
            "n_correct": sum(1 for r in fields if r["correct"]),
            "n_wrong": sum(1 for r in fields if not r["correct"]),
        }
        for case in ("new>E4a", "new<E4a", "reference"):
            sel = [r for r in allv if r["case"] == case]
            summary[t][f"teacher_top20_iou_{case}"] = mean(sel, "teacher_top20_iou")
    (rep / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (rep / "cases.md").write_text("\n".join(md), encoding="utf-8")
    _figures(summary, profiles, rep, tags)
    print(json.dumps({t: {"all_values": s["all_values"], "wrong_fields": s["wrong_fields"],
                          "n_wrong": s["n_wrong"]} for t, s in summary.items()}, indent=1))
    print(f"-> {rep}")


def _figures(summary: dict, profiles: dict, rep: Path, tags: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    students = [t for t in tags if t != "teacher"]
    cases = [("reference", "10 reference docs"), ("new>E4a", "new runs better than E4a"), ("new<E4a", "new runs worse")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    width = 0.8 / len(students)
    for k, t in enumerate(students):
        xs = [i + (k - (len(students) - 1) / 2) * width for i in range(len(cases))]
        vals = [summary[t].get(f"teacher_top20_iou_{c}") or 0.0 for c, _ in cases]
        ax.bar(xs, vals, width=width * 0.92, color=COLORS[t], label=MODELS[t][0], zorder=2)
    ax.set_xticks(range(len(cases)), [lbl for _, lbl in cases])
    ax.set_ylabel("top-20% IoU with teacher map\n(all values)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("How closely each student looks where the teacher looks", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(rep / "teacher_agreement.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for t in tags:
        if not profiles[t]:
            continue
        arr = np.stack(profiles[t])
        depth = np.arange(arr.shape[1]) / max(1, arr.shape[1] - 1)
        ax.plot(depth, arr.mean(0), color=COLORS[t], linewidth=2, label=MODELS[t][0])
    ax.set_xlabel("relative decoder depth")
    ax.set_ylabel("raw attention on image\n(visual sinks excluded)", color=INK2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(rep / "layer_profile.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
