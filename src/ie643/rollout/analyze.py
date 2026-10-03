"""Per-field rollout maps, scalar metrics and report figures.

Maps from different models live on different token grids (Qwen2.5-VL uses
28 px tokens, Qwen3-VL uses 32 px tokens on a slightly re-resized image). Both
grids cover the whole packed image, so they are compared after bilinear
resampling onto one shared grid (packed size / 28).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ie643.eval.match import EXTRACTION_FIELDS, LIST_FIELDS, META_FIELDS, is_empty, values_match
from ie643.rollout.spans import json_leaves, leaf_token_rows

# Fields shown in per-document panels (when the value is non-null in a model's output).
PANEL_FIELDS = ("vendor", "date", "total")
COMPARE_PX = 28


def safe_name(uid: str) -> str:
    return uid.replace("::", "__").replace(":", "_").replace("/", "_")


def load_doc(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        doc = {k: z[k] for k in z.files}
    doc["raw"] = str(doc["raw"])
    return doc


def field_rows(doc: dict[str, Any]) -> dict[str, list[int]]:
    """Output-token rows per field group. ``__all__`` = every non-null value token."""
    offsets = [tuple(o) for o in doc["offsets"]]
    groups: dict[str, list[int]] = {"__all__": []}
    for leaf in json_leaves(doc["raw"]):
        if leaf.is_null or leaf.end <= leaf.start:
            continue
        rows = leaf_token_rows(leaf, offsets)
        groups.setdefault(leaf.group, []).extend(rows)
        if leaf.group not in META_FIELDS:  # document_type / language are not printed values
            groups["__all__"].extend(rows)
    return {k: sorted(set(v)) for k, v in groups.items() if v}


def image_map(doc: dict[str, Any], rows: list[int], key: str = "rollout") -> np.ndarray:
    """Mean rollout over ``rows`` on image tokens, reshaped to the token grid (not normalised)."""
    r = doc[key][rows].astype(np.float32).mean(0)
    gh, gw = (int(x) for x in doc["grid"])
    return r[doc["img_pos"]].reshape(gh, gw)


def mass_split(doc: dict[str, Any], rows: list[int], key: str = "rollout") -> dict[str, float]:
    r = doc[key][rows].astype(np.float32).mean(0)
    n_prompt = int(doc["n_prompt"])
    img = np.zeros(r.shape[0], dtype=bool)
    img[doc["img_pos"]] = True
    prompt_text = np.zeros_like(img)
    prompt_text[1:n_prompt] = True
    prompt_text &= ~img
    return {
        "image": float(r[img].sum()),
        "prompt_text": float(r[prompt_text].sum()),
        "generated": float(r[n_prompt:].sum()),
        # raw (pre-removal) attention shares, averaged over layers and output tokens
        "raw_prefix_sink": float(doc["layer_prefix"].mean()),
        "raw_visual_sink": float(doc["layer_img_sink"].mean()),
        "raw_image": float(doc["layer_img"].mean()),
        "n_visual_sinks": int((~doc["keep"][doc["img_pos"]]).sum()),
    }


def concentration(m: np.ndarray) -> dict[str, float]:
    p = m.flatten()
    p = p[p > 0]  # excluded visual-sink cells are exactly 0
    p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(p.size))
    cum = np.cumsum(np.sort(p)[::-1])
    area50 = float((np.searchsorted(cum, 0.5) + 1) / p.size)
    return {"entropy_norm": ent, "area50": area50}


def resample(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(m.astype(np.float32), mode="F").resize((shape[1], shape[0]), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float64)
    return out / out.sum()


def _rank(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.size)
    r[np.argsort(x, kind="stable")] = np.arange(x.size)
    return r


def agreement(a: np.ndarray, b: np.ndarray, *, top: float = 0.2) -> dict[str, float]:
    a, b = a.flatten(), b.flatten()
    spearman = float(np.corrcoef(_rank(a), _rank(b))[0, 1])
    k = max(1, int(round(top * a.size)))
    ta, tb = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
    m = 0.5 * (a + b)
    js = 0.5 * (a * np.log((a + 1e-12) / (m + 1e-12))).sum() + 0.5 * (b * np.log((b + 1e-12) / (m + 1e-12))).sum()
    return {"spearman": spearman, "top20_iou": len(ta & tb) / len(ta | tb), "js_div": float(js)}


def field_correct(pred: dict | None, gold: dict | None, field: str) -> bool | None:
    """Exact-field correctness vs source_gt for scalar fields the GT fills; None if unscored."""
    if field not in EXTRACTION_FIELDS or field in LIST_FIELDS:
        return None
    g = (gold or {}).get(field)
    if is_empty(g):
        return None
    return bool(values_match(field, (pred or {}).get(field), g))


# ---------------------------------------------------------------- figures

STUDENT_C, TEACHER_C = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated default palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_STOPS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # blue ramp 100 -> 700


def _style(plt) -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    })


def _overlay_cmap():
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    stops = [to_rgba(c) for c in SEQ_STOPS]
    alphas = np.linspace(0.0, 0.85, len(stops))
    return LinearSegmentedColormap.from_list("rollout", [(*c[:3], a) for c, a in zip(stops, alphas)])


def doc_figure(
    image_path: Path,
    rows_by_model: dict[str, list[tuple[str, np.ndarray | None, str]]],
    out: Path,
    title: str,
    sinks: dict[str, tuple[np.ndarray, tuple[int, int]]] | None = None,
) -> None:
    """rows_by_model: model label -> list of (field, map or None, caption).
    sinks: model label -> (flat indices of excluded visual-sink tokens, token grid)."""
    import matplotlib.pyplot as plt

    _style(plt)
    img = Image.open(image_path).convert("L")
    W, H = img.size
    models = list(rows_by_model)
    fields = [f for f, _, _ in rows_by_model[models[0]]]
    cmap = _overlay_cmap()
    aspect = H / W
    fig, axes = plt.subplots(
        len(models), len(fields) + 1, figsize=(1.9 * (len(fields) + 1), 1.9 * aspect * len(models) + 0.6),
        squeeze=False,
    )
    for i, model in enumerate(models):
        ax = axes[i][0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_ylabel(model, color=INK, fontsize=10)
        ax.set_title("input" if i == 0 else "")
        for j, (field, m, cap) in enumerate(rows_by_model[model], start=1):
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.55)
            if m is not None:
                up = resample(m, (H, W))
                ax.imshow(up / up.max(), cmap=cmap, vmin=0, vmax=1, extent=(0, W, H, 0), interpolation="nearest")
                if sinks and model in sinks and len(sinks[model][0]):
                    idx, (gh, gw) = sinks[model]
                    ax.scatter((idx % gw + 0.5) * W / gw, (idx // gw + 0.5) * H / gh, s=22, marker="x",
                               color="#e34948", linewidths=1.0)
            else:
                ax.text(W / 2, H / 2, "null / absent", ha="center", va="center", color=INK2, fontsize=8)
            ax.set_title(field if i == 0 else "")
            ax.set_xlabel(cap, fontsize=7, color=INK2)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def bars_figure(stats: dict[str, dict[str, list[float]]], metric_label: str, out: Path, *, pct: bool = True) -> None:
    """stats: field -> model -> per-doc values. Horizontal grouped bars, mean with per-doc dots."""
    import matplotlib.pyplot as plt

    _style(plt)
    fields = list(stats)
    models = ["student", "teacher"]
    colors = {"student": STUDENT_C, "teacher": TEACHER_C}
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(fields) + 1.1))
    h = 0.36
    for k, model in enumerate(models):
        ys = np.arange(len(fields)) + (k - 0.5) * (h + 0.04)
        means = [np.mean(stats[f][model]) if stats[f][model] else np.nan for f in fields]
        ax.barh(ys, means, height=h, color=colors[model], label=model)
        for y, f in zip(ys, fields):
            vals = stats[f][model]
            ax.scatter(vals, [y] * len(vals), s=9, color=INK, alpha=0.45, zorder=3, linewidths=0)
        for y, v, f in zip(ys, means, fields):
            if not np.isnan(v):
                x = max([v, *stats[f][model]])  # right of the per-doc dots
                ax.text(x, y, f"  {v:.0%}" if pct else f"  {v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(np.arange(len(fields)), [f.replace("__all__", "all values") for f in fields])
    ax.invert_yaxis()
    ax.set_xlabel(metric_label)
    if pct:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(fields)), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def layer_figure(profiles: dict[str, list[np.ndarray]], out: Path) -> None:
    """Raw (non-rolled) per-layer attention mass on non-sink image tokens vs relative depth."""
    import matplotlib.pyplot as plt

    _style(plt)
    colors = {"student": STUDENT_C, "teacher": TEACHER_C}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for model, curves in profiles.items():
        if not curves:
            continue
        arr = np.stack(curves)
        depth = np.arange(arr.shape[1]) / max(1, arr.shape[1] - 1)
        for c in arr:
            ax.plot(depth, c, color=colors[model], alpha=0.15, linewidth=1)
        ax.plot(depth, arr.mean(0), color=colors[model], linewidth=2, label=f"{model} ({arr.shape[1]} layers)")
    ax.set_xlabel("relative decoder depth (0 = first layer)")
    ax.set_ylabel("raw attention on image tokens\n(visual sinks excluded)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- report

def build_report(
    manifest_rows: dict[str, dict], root: Path, run_dir: Path, models: dict[str, str], *, key: str = "rollout"
) -> dict[str, Any]:
    """models: label -> (subdir under run_dir). Expects '<label>' in {'student', 'teacher'}."""
    pred_key = {"student": "student_pred", "teacher": "teacher_gt"}
    out_dir = run_dir / ("report" if key == "rollout" else f"report_{key}")
    (out_dir / "docs").mkdir(parents=True, exist_ok=True)
    docs: dict[str, dict[str, dict]] = {}
    for label, sub in models.items():
        for path in sorted((run_dir / sub).glob("*.npz")):
            d = load_doc(path)
            docs.setdefault(str(d["uid"]), {})[label] = d

    table: list[dict[str, Any]] = []
    agree: list[dict[str, Any]] = []
    img_share: dict[str, dict[str, list[float]]] = {}
    area: dict[str, dict[str, list[float]]] = {}
    profiles: dict[str, list[np.ndarray]] = {m: [] for m in models}
    for uid, per_model in sorted(docs.items()):
        rec = manifest_rows[uid]
        H, W = int(rec["height"]), int(rec["width"])
        shared = (H // COMPARE_PX, W // COMPARE_PX)
        maps: dict[str, dict[str, np.ndarray]] = {}
        for label, d in per_model.items():
            profiles[label].append(d["layer_img"] - d["layer_img_sink"])
            maps[label] = {}
            for field, rows in field_rows(d).items():
                m = image_map(d, rows, key)
                maps[label][field] = m
                row = {
                    "uid": uid, "dataset": rec["dataset"], "model": label, "field": field, "n_tokens": len(rows),
                    "argmax_match": float(d["argmax_match"]),
                    "correct": field_correct(rec.get(pred_key[label]), rec.get("source_gt"), field),
                    **mass_split(d, rows, key), **concentration(m),
                }
                table.append(row)
                if field in ("__all__", *PANEL_FIELDS, "line_items.name", "line_items.amount", "subtotal", "tax"):
                    img_share.setdefault(field, {k: [] for k in ("student", "teacher")})[label].append(row["image"])
                    area.setdefault(field, {k: [] for k in ("student", "teacher")})[label].append(row["area50"])
        if {"student", "teacher"} <= maps.keys():
            for field in sorted(maps["student"].keys() & maps["teacher"].keys()):
                a = resample(maps["student"][field], shared)
                b = resample(maps["teacher"][field], shared)
                agree.append({"uid": uid, "dataset": rec["dataset"], "field": field, **agreement(a, b)})

        panel: dict[str, list] = {}
        for label in ("student", "teacher"):
            if label not in per_model:
                continue
            cells = []
            for field in ("__all__", *PANEL_FIELDS):
                m = maps[label].get(field)
                val = (rec.get(pred_key[label]) or {}).get(field) if field != "__all__" else None
                ok = field_correct(rec.get(pred_key[label]), rec.get("source_gt"), field)
                cap = "" if field == "__all__" else (str(val)[:26] if val is not None else "")
                if ok is not None:
                    cap += " ✓" if ok else " ✗"
                cells.append((field.replace("__all__", "all values"), m, cap))
            panel[label] = cells
        sinks = {
            label: (np.nonzero(~d["keep"][d["img_pos"]])[0], tuple(int(x) for x in d["grid"]))
            for label, d in per_model.items()
        }
        doc_figure(root / rec["image"], panel, out_dir / "docs" / f"{safe_name(uid)}.png",
                   f"{uid}    (× = visual attention sink, excluded)", sinks)

    order = [f for f in ("__all__", *PANEL_FIELDS, "line_items.name", "line_items.amount", "subtotal", "tax") if f in img_share]
    bars_figure({f: img_share[f] for f in order}, "share of rollout relevance on image tokens", out_dir / "image_share.png")
    bars_figure({f: area[f] for f in order}, "fraction of image tokens holding 50% of image relevance (lower = sharper)",
                out_dir / "area50.png")
    layer_figure(profiles, out_dir / "layer_profile.png")

    with (out_dir / "field_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    if agree:
        with (out_dir / "agreement.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agree[0]))
            w.writeheader()
            w.writerows(agree)

    def mean(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary: dict[str, Any] = {"n_docs": len(docs), "rollout": key, "models": {}}
    for label in models:
        mine = [r for r in table if r["model"] == label and r["field"] == "__all__"]
        summary["models"][label] = {k: mean(mine, k) for k in
                                    ("argmax_match", "image", "prompt_text", "raw_prefix_sink", "raw_visual_sink", "raw_image",
                                     "n_visual_sinks", "entropy_norm", "area50")}
    for label in models:
        scored = [r for r in table if r["model"] == label and r["correct"] is not None]
        summary["models"][label]["image_share_correct"] = mean([r for r in scored if r["correct"]], "image")
        summary["models"][label]["image_share_wrong"] = mean([r for r in scored if not r["correct"]], "image")
        summary["models"][label]["n_scored_fields"] = len(scored)
        summary["models"][label]["n_wrong_fields"] = sum(1 for r in scored if not r["correct"])
    if agree:
        summary["agreement_all_values"] = {k: mean([a for a in agree if a["field"] == "__all__"], k)
                                           for k in ("spearman", "top20_iou", "js_div")}
        summary["agreement_by_field"] = {
            f: {k: mean([a for a in agree if a["field"] == f], k) for k in ("spearman", "top20_iou")}
            for f in order
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
