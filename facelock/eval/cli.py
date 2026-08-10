"""``facelock-eval`` console script (T4) -- offline biometric accuracy tooling.

Two verbs, both offline w.r.t. the daemon (no camera takeover, no systemd):

  report  Score a genuine-probe npz + an impostor npz through the DEPLOYED
          matcher (live ``cfg.recognition.pose_max`` / metric) and emit a JSON
          FMR/FNMR/EER report (+ optional DET/ROC PNGs). This is the number that
          closes the audit gap (protocol §0, D-1).

  embed   Turn an image directory or the LFW dataset into an anonymous impostor
          embedding npz through the SAME YuNet+SFace pipeline the daemon uses
          (embeddings ONLY -- pixels discarded, REQ-NF-13). Refuses to mix a
          model different from the enrolled template's.

Import safety: cv2 / sklearn / matplotlib are all imported lazily inside the
verb bodies (via the eval modules), so ``--help`` and the ``report`` path work
with no OpenCV. The genuine-capture step (protocol T5) needs the user + a camera
and is intentionally NOT part of this CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import paths as _paths
from ..config import Config, load_config
from ..errors import ConfigError, ModelError, TemplateError
from ..store import TemplateStore
from . import embed_dataset as ED
from . import report as report


def _load_cfg(config_path: Path | None) -> Config:
    return load_config(config_path).resolve_model_paths(_paths.models_dir())


def _print_config_errors(exc: Exception) -> None:
    print(f"config error: {exc}", file=sys.stderr)
    for err in getattr(exc, "errors", []) or []:
        print(f"  - {err}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cmd_report(args: argparse.Namespace) -> int:
    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2

    store = TemplateStore(args.template) if args.template else TemplateStore()
    try:
        template = store.load()
    except TemplateError as exc:
        print(f"report: cannot load owner template: {exc}", file=sys.stderr)
        return 3

    try:
        genuine, gen_meta = ED.load_embeddings_npz(args.genuine)
        impostor, imp_meta = ED.load_embeddings_npz(args.impostor)
    except (OSError, ValueError) as exc:
        print(f"report: cannot load embedding file: {exc}", file=sys.stderr)
        return 4

    if genuine.shape[0] == 0 or impostor.shape[0] == 0:
        print("report: genuine and impostor sets must both be non-empty.", file=sys.stderr)
        return 4

    # LIVE deployment knob: never a hardcoded default. --pose-max overrides.
    pose_max = args.pose_max if args.pose_max is not None else int(cfg.recognition.pose_max)

    gen_sha = ED.file_sha256(args.genuine)
    imp_sha = ED.file_sha256(args.impostor)
    n_ids = imp_meta.get("n_identities")

    out_dir = Path(args.out)
    result = report.run(
        template,
        genuine,
        impostor,
        out_dir,
        pose_max=pose_max,
        fmr_target=float(cfg.recognition.fmr_target),
        fnmr_target=float(cfg.recognition.fnmr_target),
        match_votes=int(cfg.recognition.match_votes),
        probe_frames=int(cfg.recognition.probe_frames),
        tau_floor=float(cfg.recognition.tau_floor),
        metric=template.metric,
        bootstrap=int(args.bootstrap),
        genuine_meta=gen_meta,
        impostor_meta=imp_meta,
        n_impostor_identities=int(n_ids) if n_ids is not None else None,
        genuine_sha256=gen_sha,
        impostor_sha256=imp_sha,
        make_plots=not args.no_plot,
        json_name=args.json_name,
    )

    r = result.report
    prov = r["provenance"]
    if not prov["model_consistent"]:
        print("report: WARNING -- template model_id differs from the embedding "
              "set's model_id; the eval bank may not match the deployed model.",
              file=sys.stderr)
    at = r["at_shipped_tau"]
    print(f"report written: {out_dir / args.json_name}")
    print(f"  pose_max={r['config']['pose_max']} metric={r['config']['metric']} "
          f"tau={r['config']['tau']:.4f} bank={r['template']['bank_size']}")
    print(f"  N: genuine={r['counts']['n_genuine']} impostor={r['counts']['n_impostor']}")
    print(f"  FMR@tau  = {at['fmr_frame']['rate']:.4g}  "
          f"CI[{at['fmr_frame']['ci'][0]:.4g},{at['fmr_frame']['ci'][1]:.4g}]")
    print(f"  FNMR@tau = {at['fnmr_frame']['rate']:.4g}  "
          f"CI[{at['fnmr_frame']['ci'][0]:.4g},{at['fnmr_frame']['ci'][1]:.4g}]")
    print(f"  FMR_sys ({at['fmr_sys_kofn']['k']}-of-{at['fmr_sys_kofn']['n']}) "
          f"= {at['fmr_sys_kofn']['value']:.4g}")
    print(f"  EER      = {r['eer']['value']:.4g}  "
          f"CI[{r['eer']['ci'][0]:.4g},{r['eer']['ci'][1]:.4g}]")
    if r["plot"]["skipped"]:
        print(f"  plot: skipped ({r['plot']['reason']})")
    else:
        print(f"  plot: {r['plot']['det_png']}, {r['plot']['roc_png']}")
    return 0


# --------------------------------------------------------------------------- #
# embed
# --------------------------------------------------------------------------- #
def cmd_embed(args: argparse.Namespace) -> int:
    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2

    if not args.lfw and not args.source:
        print("embed: provide --source <image-dir> or --lfw.", file=sys.stderr)
        return 2

    yunet = args.yunet or cfg.detection.model_path
    sface = args.sface or cfg.recognition.model_path
    try:
        detector, embedder, model_id = ED.build_pipeline(
            yunet, sface,
            confidence_floor=float(cfg.detection.confidence_floor),
            nms_threshold=float(cfg.detection.nms_threshold),
            min_face_px=int(cfg.detection.min_face_px),
        )
    except ModelError as exc:
        print(f"embed: {exc}\nRun scripts/download_models.sh first.", file=sys.stderr)
        return 2

    # If a template is enrolled, refuse to embed with a different model (protocol §2b).
    expected = None
    tmpl = TemplateStore().try_load()
    if tmpl is not None and getattr(tmpl, "model_id", ""):
        expected = tmpl.model_id

    try:
        if args.lfw:
            result = ED.embed_lfw(
                detector, embedder,
                min_faces_per_person=int(args.min_faces),
                one_per_identity=bool(args.one_per_identity),
                model_id=model_id, expected_model_id=expected,
                resize=float(args.resize),
            )
        else:
            result = ED.embed_image_dir(
                args.source, detector, embedder,
                dataset=args.dataset, model_id=model_id, expected_model_id=expected,
            )
    except ModelError as exc:
        print(f"embed: {exc}", file=sys.stderr)
        return 5
    except FileNotFoundError as exc:
        print(f"embed: {exc}", file=sys.stderr)
        return 4

    ED.write_embeddings_npz(result, args.out)
    print(f"embeddings written: {args.out}")
    print(json.dumps(result.provenance, indent=2, default=str))
    if result.provenance["n_valid"] == 0:
        print("embed: WARNING -- no valid single-face embeddings produced.", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facelock-eval",
        description="facelock -- offline biometric accuracy evaluation (FMR/FNMR/EER).",
    )
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    sub = parser.add_subparsers(dest="verb", required=True)

    p_report = sub.add_parser("report", help="score genuine+impostor npzs -> JSON report")
    p_report.add_argument("--template", type=Path, default=None,
                          help="owner template path (default: enrolled template)")
    p_report.add_argument("--genuine", required=True, help="genuine-probe embedding npz")
    p_report.add_argument("--impostor", required=True, help="impostor embedding npz")
    p_report.add_argument("--out", default=".", help="output directory for JSON + PNGs")
    p_report.add_argument("--json-name", default="eval_report.json", help="report filename")
    p_report.add_argument("--pose-max", type=int, default=None,
                          help="override pose_max (default: LIVE cfg.recognition.pose_max)")
    p_report.add_argument("--bootstrap", type=int, default=2000,
                          help="EER bootstrap resamples (0 disables the EER CI)")
    p_report.add_argument("--no-plot", action="store_true",
                          help="skip DET/ROC PNGs (JSON still written)")
    p_report.set_defaults(func=cmd_report)

    p_embed = sub.add_parser("embed", help="image dir / LFW -> impostor embedding npz")
    grp = p_embed.add_mutually_exclusive_group()
    grp.add_argument("--source", default=None, help="directory of face images")
    grp.add_argument("--lfw", action="store_true", help="use the LFW dataset (sklearn)")
    p_embed.add_argument("--out", required=True, help="output embedding npz path")
    p_embed.add_argument("--dataset", default="image-dir", help="dataset label for provenance")
    p_embed.add_argument("--yunet", default=None, help="YuNet model path (default: config)")
    p_embed.add_argument("--sface", default=None, help="SFace model path (default: config)")
    p_embed.add_argument("--min-faces", type=int, default=0,
                         help="LFW: min_faces_per_person filter")
    p_embed.add_argument("--one-per-identity", action="store_true",
                         help="LFW: keep one image per identity (independent estimate)")
    p_embed.add_argument("--resize", type=float, default=2.0,
                         help="LFW: fetch_lfw_people resize factor (default 2.0 -- keeps "
                              "faces above min_face_px; sklearn's own default of 0.5 "
                              "shrinks faces below the detector's gate)")
    p_embed.set_defaults(func=cmd_embed)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
