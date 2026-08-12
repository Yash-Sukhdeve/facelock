"""Accuracy report (T4): deployed FMR / FNMR / EER + operating points + CIs.

Glues the D-1 deployed-matcher scorer (:mod:`facelock.eval.scoring`) to the
ISO/IEC 19795-1 metrics (:mod:`facelock.eval.metrics`) and emits a JSON report
plus best-effort DET/ROC PNGs. Every number is a property of the system *as
shipped*, because both the genuine and impostor probe sets are scored through
:func:`facelock.eval.scoring.score_probes` at the **LIVE** ``pose_max`` / metric
-- the same max-over-bank the daemon runs. Threading the deployed ``pose_max``
here is load-bearing: a default of 5 while the install uses a different
``pose_max`` would build a different eval bank and *understate* FMR -- exactly
the audit error this harness exists to prevent. The JSON records the
``pose_max`` / ``metric`` / ``tau`` actually used so the reported number is
traceable.

Reported quantities (protocol §1, §3):
  * FMR / FNMR at the SHIPPED tau, each with a Wilson 95% CI [C2];
  * per-decision system FMR under k-of-n voting (``system_rate_kofn``);
  * EER with a seeded bootstrap CI;
  * FNMR @ FMR = 1e-2 (headline REQ-NF-10) and FMR @ FNMR = 5% (ASM-05),
    each with a Wilson CI built from the RAW success count (A1 fix).

Offline + import-safe: no cv2, no models, no network, no daemon. matplotlib is
imported lazily and only for the optional plot; if it is absent the JSON is
still written and the plot is marked skipped (never a failure).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths as _paths
from . import metrics as metrics
from . import scoring as scoring

EMBEDDING_DIM = 128
TAU_FLOOR_DEFAULT = 0.363          # SFace published cosine operating point (calibrate.py)
SCHEMA_VERSION = 1

__all__ = [
    "EvalResult",
    "build_report",
    "write_plots",
    "write_json",
    "run",
]


@dataclass
class EvalResult:
    """A built report plus the deployed scores + DET curve it was computed from.

    ``report`` is the JSON-serialisable dict (scalars only). The score arrays and
    ``det`` tuple are retained ONLY in-process to draw the DET/ROC plot; they are
    never written to the JSON (which stays compact and pixel-free).
    """

    report: dict[str, Any]
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray
    det: tuple[np.ndarray, np.ndarray, np.ndarray]
    higher_is_better: bool


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_matrix(probes) -> np.ndarray:
    arr = np.asarray(probes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.reshape(-1, EMBEDDING_DIM)


def build_report(
    template,
    genuine,
    impostor,
    *,
    pose_max: int,
    fmr_target: float = 0.01,
    fnmr_target: float = 0.05,
    match_votes: int = 3,
    probe_frames: int = 5,
    tau_floor: float = TAU_FLOOR_DEFAULT,
    metric: str | None = None,
    bootstrap: int = 2000,
    seed: int = 20260730,
    genuine_meta: dict[str, Any] | None = None,
    impostor_meta: dict[str, Any] | None = None,
    n_impostor_identities: int | None = None,
    genuine_sha256: str | None = None,
    impostor_sha256: str | None = None,
) -> EvalResult:
    """Compute the full deployed-accuracy report for one owner template.

    ``pose_max`` MUST be the live ``cfg.recognition.pose_max`` -- it is threaded
    verbatim into :func:`scoring.score_probes` for BOTH probe sets so the eval
    bank equals the deployed bank. ``metric`` defaults to ``template.metric``.
    """
    metric = metric or getattr(template, "metric", "cosine")
    hib = metrics.higher_is_better_for_metric(metric)

    gen = _as_matrix(genuine)
    imp = _as_matrix(impostor)

    # Fail CLEARLY on an empty probe set instead of letting an empty impostor
    # array fall through to `calibrate._tau_at_fmr_cosine`, which indexes
    # `scores[-1]` on a zero-length sorted array (IndexError) -- e.g. every LFW
    # image dropped by the one-face gate because `embed_lfw`'s `resize` shrank
    # faces below `min_face_px`. Checked BEFORE any scoring/metric call so the
    # failure is diagnosable from the message alone, not a stack trace.
    if imp.shape[0] == 0:
        raise ValueError(
            "impostor set is empty -- check min_face_px vs image resize "
            "(embed_lfw's `resize` must keep faces above the detector's "
            "min_face_px gate; the default resize=2.0 does this for LFW)"
        )
    if gen.shape[0] == 0:
        raise ValueError(
            "genuine set is empty -- check min_face_px vs image resize/capture "
            "quality (the one-face gate dropped every genuine probe image)"
        )

    # --- D-1: score through the EXACT deployed matcher at the LIVE pose_max. ---
    gen_scores = scoring.score_probes(template, gen, pose_max=pose_max)
    imp_scores = scoring.score_probes(template, imp, pose_max=pose_max)

    tau_shipped = float(template.tau)

    # --- Rates at the shipped tau (raw counts -> exact Wilson CIs, A1 fix). ---
    n_gen = int(gen_scores.size)
    n_imp = int(imp_scores.size)
    fmr_count = metrics.fmr_count_at_tau(imp_scores, tau_shipped, hib)
    fmr_rate = 0.0 if n_imp == 0 else fmr_count / n_imp
    fmr_ci = metrics.wilson(fmr_count, n_imp)
    fnmr_count = metrics.fnmr_count_at_tau(gen_scores, tau_shipped, hib)
    fnmr_rate = 0.0 if n_gen == 0 else fnmr_count / n_gen
    fnmr_ci = metrics.wilson(fnmr_count, n_gen)
    fmr_sys = metrics.system_rate_kofn(fmr_rate, match_votes, probe_frames)

    # --- EER (interpolated crossing + seeded bootstrap CI). ---
    eer = metrics.eer(gen_scores, imp_scores, hib, bootstrap=bootstrap, seed=seed)

    # --- Requirement-aligned operating points. ---
    op_fnmr = metrics.fnmr_at_fmr(gen_scores, imp_scores, fmr_target, hib)   # FNMR @ FMR<=1e-2
    op_fnmr_count = metrics.fnmr_count_at_tau(gen_scores, op_fnmr.tau, hib)
    op_fmr = metrics.fmr_at_fnmr(gen_scores, imp_scores, fnmr_target, hib)   # FMR  @ FNMR<=5%
    op_fmr_count = metrics.fmr_count_at_tau(imp_scores, op_fmr.tau, hib)

    det = metrics.det_points(gen_scores, imp_scores, hib)

    # Bank the scorer actually built at this pose_max (matches deployment).
    matcher = scoring.build_matcher(template, pose_max=pose_max)
    bank_size = int(matcher.pose_count)
    n_samples = int(np.asarray(template.samples).reshape(-1, EMBEDDING_DIM).shape[0])

    template_model_id = str(getattr(template, "model_id", "") or "")
    imp_model_id = str((impostor_meta or {}).get("model_id", "") or "")
    gen_model_id = str((genuine_meta or {}).get("model_id", "") or "")
    model_consistent = True
    for other in (imp_model_id, gen_model_id):
        if template_model_id and other and other != template_model_id:
            model_consistent = False

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "facelock-eval",
        "generated_at": _now_iso(),
        "config": {
            "pose_max": int(pose_max),
            "metric": metric,
            "tau": tau_shipped,
            "tau_floor": float(tau_floor),
            "tau_above_floor": bool(tau_shipped >= float(tau_floor)) if hib else None,
            "fmr_target": float(fmr_target),
            "fnmr_target": float(fnmr_target),
            "match_votes_k": int(match_votes),
            "probe_frames_n": int(probe_frames),
            "higher_is_better": bool(hib),
            "bootstrap": int(bootstrap),
            "seed": int(seed),
        },
        "template": {
            "owner_name": str(getattr(template, "owner_name", "")),
            "model_id": template_model_id,
            "bank_size": bank_size,
            "n_samples": n_samples,
            "metric": metric,
            "tau": tau_shipped,
        },
        "counts": {
            "n_genuine": n_gen,
            "n_impostor": n_imp,
            "n_impostor_identities": (
                int(n_impostor_identities) if n_impostor_identities is not None else None
            ),
            "n_impostor_images": int((impostor_meta or {}).get("n_valid", n_imp)),
        },
        "at_shipped_tau": {
            "tau": tau_shipped,
            "fmr_frame": {
                "rate": float(fmr_rate),
                "count": int(fmr_count),
                "n": n_imp,
                "ci": [float(fmr_ci[0]), float(fmr_ci[1])],
            },
            "fnmr_frame": {
                "rate": float(fnmr_rate),
                "count": int(fnmr_count),
                "n": n_gen,
                "ci": [float(fnmr_ci[0]), float(fnmr_ci[1])],
            },
            "fmr_sys_kofn": {
                "k": int(match_votes),
                "n": int(probe_frames),
                "value": float(fmr_sys),
            },
        },
        "eer": {
            "value": float(eer.value),
            "tau": float(eer.tau),
            "ci": [float(eer.ci[0]), float(eer.ci[1])],
            "bootstrap": int(bootstrap),
        },
        "operating_points": {
            "fnmr_at_fmr_1e-2": {
                "fmr_target": float(fmr_target),
                "tau": float(op_fnmr.tau),
                "rate": float(op_fnmr.rate),
                "fnmr": float(op_fnmr.rate),
                "count": int(op_fnmr_count),
                "n": n_gen,
                "ci": [float(op_fnmr.ci[0]), float(op_fnmr.ci[1])],
            },
            "fmr_at_fnmr_5pct": {
                "fnmr_target": float(fnmr_target),
                "tau": float(op_fmr.tau),
                "rate": float(op_fmr.rate),
                "fmr": float(op_fmr.rate),
                "count": int(op_fmr_count),
                "n": n_imp,
                "ci": [float(op_fmr.ci[0]), float(op_fmr.ci[1])],
            },
        },
        "provenance": {
            "genuine": genuine_meta or {},
            "impostor": impostor_meta or {},
            "dataset": {
                "name": str((impostor_meta or {}).get("dataset", "")),
                "version": str((impostor_meta or {}).get("dataset_version", "")),
                "sha256": impostor_sha256 or "",
            },
            "genuine_sha256": genuine_sha256 or "",
            "model_id": template_model_id,
            "model_consistent": bool(model_consistent),
        },
        # Filled by run(); a bare build_report() leaves it un-plotted.
        "plot": {"det_png": None, "roc_png": None, "skipped": True,
                 "reason": "plot not requested (build_report only)"},
    }
    return EvalResult(
        report=report,
        genuine_scores=gen_scores,
        impostor_scores=imp_scores,
        det=det,
        higher_is_better=hib,
    )


# --------------------------------------------------------------------------- #
# DET / ROC plot -- matplotlib gated (optional; best-effort).
# --------------------------------------------------------------------------- #
def _import_pyplot():
    """Return a headless (Agg) ``pyplot`` module, or ``None`` if unavailable.

    Isolated so tests can force the absent-path deterministically by patching
    this one function.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def write_plots(result: EvalResult, out_dir: str | Path, *, prefix: str = "eval") -> dict[str, Any]:
    """Write DET + ROC PNGs; skip gracefully (no error) if matplotlib is absent."""
    plt = _import_pyplot()
    if plt is None:
        return {
            "det_png": None,
            "roc_png": None,
            "skipped": True,
            "reason": "matplotlib unavailable -- plot skipped (JSON still written)",
        }
    taus, fmr, fnmr = result.det
    out_dir = Path(out_dir)
    _paths.ensure_dir(out_dir, 0o700)
    det_path = out_dir / f"{prefix}_det.png"
    roc_path = out_dir / f"{prefix}_roc.png"

    # DET: FNMR vs FMR (ISO/IEC 19795-1 primary characterization [C1]).
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fmr, fnmr, marker=".", linewidth=1)
    ax.set_xlabel("FMR (false match rate)")
    ax.set_ylabel("FNMR (false non-match rate)")
    ax.set_title("DET -- deployed matcher")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(det_path, dpi=120)
    plt.close(fig)

    # ROC: TAR = 1 - FNMR vs FMR.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fmr, 1.0 - fnmr, marker=".", linewidth=1)
    ax.set_xlabel("FMR (false match rate)")
    ax.set_ylabel("TAR = 1 - FNMR")
    ax.set_title("ROC -- deployed matcher")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    return {"det_png": str(det_path), "roc_png": str(roc_path), "skipped": False, "reason": None}


def write_json(report: dict[str, Any], path: str | Path) -> Path:
    """Write the report dict as pretty JSON (0700 parent dir)."""
    path = Path(path)
    _paths.ensure_dir(path.parent, 0o700)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def run(
    template,
    genuine,
    impostor,
    out_dir: str | Path,
    *,
    pose_max: int,
    make_plots: bool = True,
    json_name: str = "eval_report.json",
    plot_prefix: str = "eval",
    **kwargs: Any,
) -> EvalResult:
    """Build the report, (optionally) draw plots, and write the JSON artefact.

    Always writes the JSON. Plots are best-effort: disabled by ``make_plots=False``
    or when matplotlib is absent -- either way ``report['plot']`` records the
    outcome so the artefact is self-describing.
    """
    out_dir = Path(out_dir)
    result = build_report(template, genuine, impostor, pose_max=pose_max, **kwargs)
    if make_plots:
        plot_info = write_plots(result, out_dir, prefix=plot_prefix)
    else:
        plot_info = {"det_png": None, "roc_png": None, "skipped": True,
                     "reason": "plots disabled (--no-plot)"}
    result.report["plot"] = plot_info
    write_json(result.report, out_dir / json_name)
    return result
