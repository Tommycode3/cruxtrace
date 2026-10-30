"""Event-level and frame-level evaluation of climb segmentation."""

from __future__ import annotations

import numpy as np


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_events(gt, pred, thr: float = 0.5):
    """Greedy one-to-one matching by descending IoU; a match needs IoU >= thr."""
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)), reverse=True)
    used_g, used_p, matches = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matches.append((i, j, v))
    return matches


def event_scores(gt, pred, thr: float = 0.5) -> dict:
    m = match_events(gt, pred, thr)
    tp = len(m)
    prec = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    rec = tp / len(gt) if gt else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ious = [v for *_, v in m]
    starts = [abs(gt[i][0] - pred[j][0]) for i, j, _ in m]
    ends = [abs(gt[i][1] - pred[j][1]) for i, j, _ in m]
    return {
        "tp": tp, "n_gt": len(gt), "n_pred": len(pred),
        "precision": prec, "recall": rec, "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "count_error": len(pred) - len(gt),
        "start_mae_s": float(np.mean(starts)) if starts else float("nan"),
        "end_mae_s": float(np.mean(ends)) if ends else float("nan"),
    }


def frame_scores(gt, pred, duration: float, dt: float = 0.1) -> dict:
    t = np.arange(0, duration, dt)

    def mask(iv):
        m = np.zeros(len(t), bool)
        for a, b in iv:
            m |= (t >= a) & (t < b)
        return m

    g, p = mask(gt), mask(pred)
    tp, fp, fn = np.sum(g & p), np.sum(~g & p), np.sum(g & ~p)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return {
        "frame_precision": float(prec), "frame_recall": float(rec),
        "frame_f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "time_on_wall_gt_s": float(g.sum() * dt), "time_on_wall_pred_s": float(p.sum() * dt),
    }
