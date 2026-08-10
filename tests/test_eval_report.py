"""End-to-end report + CLI tests (T4).

The report glues T2 (deployed-matcher scoring) to T1 (metrics) and emits a JSON
accuracy report plus best-effort DET/ROC PNGs. These tests run entirely offline
on fixture embedding matrices -- no camera, model, daemon, or network -- and pin
the two properties the A1 review demanded:

  * the LIVE ``pose_max`` (and ``metric``) is threaded into ``score_probes`` so
    the eval bank matches the deployed matcher exactly
    (``test_report_threads_pose_max_into_scorer`` +
     ``test_cli_report_uses_live_config_pose_max``);
  * the JSON records ``pose_max`` / ``metric`` / ``tau`` actually used, so every
    reported number is traceable.

matplotlib is optional: with it absent the report still emits JSON and records
that the plot was skipped (never fails).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from facelock.calibrate import centroid_of
from facelock.eval import embed_dataset as ED
from facelock.eval import report as report
from facelock.eval import scoring as scoring
from facelock.store import Template, TemplateStore, generate_synthetic_impostors
from tests.conftest import owner_cluster, unit_vec


# --------------------------------------------------------------------------- #
# Fixtures: an owner template + held-out genuine probes + impostor embeddings.
# --------------------------------------------------------------------------- #
def make_template(tau: float = 0.3, n: int = 8, jitter: float = 0.08,
                  seed: int = 1, model_id: str = "deadbeef") -> Template:
    samples = owner_cluster(n=n, seed=seed, jitter=jitter)
    return Template(
        owner_name="Yash",
        centroid=centroid_of(samples),
        samples=samples,
        tau=tau,
        metric="cosine",
        model_id=model_id,
    )


def genuine_probes(m: int = 40, seed: int = 1, jitter: float = 0.08) -> np.ndarray:
    """Held-out owner probes: same direction as enrollment, fresh jitter (NOT in the bank)."""
    base = unit_vec(seed)
    rng = np.random.default_rng(4242)
    out = []
    for _ in range(m):
        v = base + jitter * rng.standard_normal(base.shape).astype(np.float32)
        out.append((v / np.linalg.norm(v)).astype(np.float32))
    return np.stack(out)


def _tau_for_partial_fmr(template: Template, impostors: np.ndarray, q: float = 0.8) -> float:
    """A tau at the q-quantile of deployed impostor scores -> ~ (1-q) FMR at tau."""
    scores = scoring.score_probes(template, impostors, pose_max=5)
    return float(np.quantile(scores, q))


# --------------------------------------------------------------------------- #
# Report schema + values.
# --------------------------------------------------------------------------- #
def test_build_report_schema_and_values():
    tmpl0 = make_template(tau=0.363, n=8)
    imp = generate_synthetic_impostors(n=600, seed=42)
    tau = _tau_for_partial_fmr(tmpl0, imp, q=0.8)      # ~20% impostors accept
    tmpl = make_template(tau=tau, n=8)
    gen = genuine_probes(m=40)

    res = report.build_report(tmpl, gen, imp, pose_max=5, match_votes=3,
                              probe_frames=5, bootstrap=200)
    r = res.report

    for key in ("generated_at", "config", "template", "counts",
                "at_shipped_tau", "eer", "operating_points", "provenance", "plot"):
        assert key in r, f"missing report key {key}"

    # Config -- the traceable, deployed knobs.
    assert r["config"]["pose_max"] == 5
    assert r["config"]["metric"] == "cosine"
    assert r["config"]["tau"] == pytest.approx(tmpl.tau)
    assert r["config"]["tau_floor"] == pytest.approx(0.363)

    # Counts.
    assert r["counts"]["n_genuine"] == gen.shape[0]
    assert r["counts"]["n_impostor"] == imp.shape[0]

    # FMR at the shipped tau lies inside its own Wilson CI, and is a real fraction.
    at = r["at_shipped_tau"]
    fmr = at["fmr_frame"]
    assert fmr["ci"][0] <= fmr["rate"] <= fmr["ci"][1]
    assert 0.0 < fmr["rate"] < 0.5
    assert fmr["count"] == pytest.approx(round(fmr["rate"] * imp.shape[0]))

    # k-of-n voting makes the SYSTEM false-unlock rate strictly rarer per-frame.
    assert at["fmr_sys_kofn"]["value"] < fmr["rate"]
    assert at["fmr_sys_kofn"]["k"] == 3 and at["fmr_sys_kofn"]["n"] == 5

    # Operating points present and requirement-aligned.
    op = r["operating_points"]
    assert op["fnmr_at_fmr_1e-2"]["fmr_target"] == pytest.approx(0.01)
    assert op["fmr_at_fnmr_5pct"]["fnmr_target"] == pytest.approx(0.05)
    for name in ("fnmr_at_fmr_1e-2", "fmr_at_fnmr_5pct"):
        assert set(("tau", "rate", "count", "n", "ci")).issubset(op[name].keys())

    # EER sane + bootstrap CI present.
    assert 0.0 <= r["eer"]["value"] <= 0.5
    assert r["eer"]["ci"][0] <= r["eer"]["ci"][1]

    # Provenance threads the model id (must equal the matcher/template's model).
    assert r["template"]["model_id"] == "deadbeef"
    assert r["template"]["bank_size"] == 1 + min(5, 8)


def test_operating_point_ci_from_exact_count_a1_fix():
    # The report's operating-point CIs come straight from the raw counts (A1 fix).
    tmpl = make_template(tau=0.3, n=8)
    gen = genuine_probes(m=60)
    imp = generate_synthetic_impostors(n=800, seed=11)
    res = report.build_report(tmpl, gen, imp, pose_max=5, bootstrap=0)
    op = res.report["operating_points"]["fnmr_at_fmr_1e-2"]
    from facelock.eval import metrics as M
    # rate == count / n exactly (no float round-trip).
    assert op["rate"] == pytest.approx(op["count"] / op["n"])
    assert tuple(op["ci"]) == M.wilson(op["count"], op["n"])


# --------------------------------------------------------------------------- #
# THE load-bearing requirement: live pose_max is threaded into the scorer.
# --------------------------------------------------------------------------- #
def test_report_threads_pose_max_into_scorer(monkeypatch):
    from facelock.eval import scoring as S

    calls: list = []
    real = S.score_probes

    def spy(template, probes, *, pose_max=None):
        calls.append(pose_max)
        return real(template, probes, pose_max=pose_max)

    monkeypatch.setattr(S, "score_probes", spy)

    NON_DEFAULT = 9                       # != scoring.DEFAULT_POSE_MAX (5)
    assert NON_DEFAULT != scoring.DEFAULT_POSE_MAX
    tmpl = make_template(tau=0.3, n=10)
    gen = genuine_probes(m=20)
    imp = generate_synthetic_impostors(n=300, seed=7)

    res = report.build_report(tmpl, gen, imp, pose_max=NON_DEFAULT, bootstrap=0)

    # BOTH the genuine and impostor probe sets are scored at the LIVE pose_max --
    # not the default 5 -- so the eval bank matches the deployed matcher exactly.
    assert calls == [NON_DEFAULT, NON_DEFAULT]
    assert res.report["config"]["pose_max"] == NON_DEFAULT
    # The bank actually built for scoring reflects the threaded pose_max.
    assert res.report["template"]["bank_size"] == 1 + min(NON_DEFAULT, 10)


def test_report_metric_threaded_for_l2_template():
    # An l2 template flips the accept direction; the report must honour it.
    samples = owner_cluster(n=6, seed=2, jitter=0.1)
    tmpl = Template(owner_name="l2", centroid=centroid_of(samples), samples=samples,
                    tau=0.9, metric="l2", model_id="m2")
    gen = genuine_probes(m=30, seed=2)
    imp = generate_synthetic_impostors(n=300, seed=3)
    res = report.build_report(tmpl, gen, imp, pose_max=4, bootstrap=0)
    assert res.report["config"]["metric"] == "l2"
    assert res.higher_is_better is False


# --------------------------------------------------------------------------- #
# matplotlib is optional: absent -> JSON still emitted, plot marked skipped.
# --------------------------------------------------------------------------- #
def test_write_plots_skipped_when_matplotlib_absent(monkeypatch, tmp_path):
    tmpl = make_template(tau=0.3)
    res = report.build_report(tmpl, genuine_probes(20),
                              generate_synthetic_impostors(200, seed=5),
                              pose_max=5, bootstrap=0)
    monkeypatch.setattr(report, "_import_pyplot", lambda: None)
    info = report.write_plots(res, tmp_path)
    assert info["skipped"] is True
    assert info["det_png"] is None and info["roc_png"] is None
    assert "matplotlib" in (info["reason"] or "").lower()


def test_run_emits_json_even_without_plots(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "_import_pyplot", lambda: None)
    tmpl = make_template(tau=0.3)
    res = report.run(tmpl, genuine_probes(20),
                     generate_synthetic_impostors(200, seed=6),
                     tmp_path, pose_max=5, bootstrap=0)
    jpath = tmp_path / "eval_report.json"
    assert jpath.exists()
    data = json.loads(jpath.read_text())
    assert data["plot"]["skipped"] is True
    assert "at_shipped_tau" in data
    assert data["config"]["pose_max"] == 5


def test_write_plots_present_writes_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    tmpl = make_template(tau=0.3)
    res = report.build_report(tmpl, genuine_probes(20),
                              generate_synthetic_impostors(200, seed=8),
                              pose_max=5, bootstrap=0)
    info = report.write_plots(res, tmp_path)
    assert info["skipped"] is False
    assert Path(info["det_png"]).exists()
    assert Path(info["roc_png"]).exists()


# --------------------------------------------------------------------------- #
# CLI: end-to-end on fixture npzs; the live-config pose_max reaches the report.
# --------------------------------------------------------------------------- #
def _write_config(pose_max: int) -> Path:
    cfg_dir = Path(os.environ["XDG_CONFIG_HOME"]) / "facelock"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.toml"
    cfg_path.write_text(
        "[recognition]\n"
        f"pose_max = {pose_max}\n"
        "[security]\n"
        'phase = "P"\n'
    )
    return cfg_path


def test_cli_report_uses_live_config_pose_max(tmp_path):
    from facelock.eval import cli as eval_cli

    _write_config(pose_max=8)                       # NON-default (default 5)
    tmpl = make_template(tau=0.3, n=10)
    TemplateStore().save(tmpl)

    gpath = tmp_path / "genuine.npz"
    ipath = tmp_path / "impostor.npz"
    ED.write_embeddings_npz(genuine_probes(30), gpath)
    ED.write_embeddings_npz(generate_synthetic_impostors(400, seed=9), ipath)

    out_dir = tmp_path / "out"
    rc = eval_cli.main([
        "report", "--genuine", str(gpath), "--impostor", str(ipath),
        "--out", str(out_dir), "--no-plot", "--bootstrap", "0",
    ])
    assert rc == 0
    jpath = out_dir / "eval_report.json"
    assert jpath.exists()
    data = json.loads(jpath.read_text())
    # The CLI must read the LIVE cfg.recognition.pose_max (8), not a hardcoded 5.
    assert data["config"]["pose_max"] == 8
    assert data["template"]["bank_size"] == 1 + min(8, 10)
    assert data["counts"]["n_genuine"] == 30
    assert data["counts"]["n_impostor"] == 400


def test_cli_report_missing_template_fails_cleanly(tmp_path):
    from facelock.eval import cli as eval_cli

    _write_config(pose_max=5)
    gpath = tmp_path / "g.npz"
    ipath = tmp_path / "i.npz"
    ED.write_embeddings_npz(genuine_probes(5), gpath)
    ED.write_embeddings_npz(generate_synthetic_impostors(150, seed=1), ipath)
    rc = eval_cli.main([
        "report", "--genuine", str(gpath), "--impostor", str(ipath),
        "--out", str(tmp_path / "out"), "--no-plot",
    ])
    assert rc != 0                                  # no template enrolled -> non-zero


def test_cli_embed_dir_end_to_end(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from facelock.eval import cli as eval_cli
    import tests.test_eval_embed as te

    # Inject fake detector/embedder so no models/camera are needed; cv2 is used
    # only to imread the synthetic frames we write to disk.
    monkeypatch.setattr(
        ED, "build_pipeline",
        lambda *a, **k: (te.FakeDetector(), te.FakeEmbedder(), "mid"),
    )
    _write_config(pose_max=5)
    src = tmp_path / "imgs"
    src.mkdir()
    cv2.imwrite(str(src / "a.png"), te.frame(1, seed=1))
    cv2.imwrite(str(src / "b.png"), te.frame(0))           # no face -> dropped
    cv2.imwrite(str(src / "c.png"), te.frame(1, seed=2))
    out = tmp_path / "imp.npz"

    rc = eval_cli.main(["embed", "--source", str(src), "--out", str(out)])
    assert rc == 0
    emb, meta = ED.load_embeddings_npz(out)
    assert emb.shape == (2, 128)
    assert meta["model_id"] == "mid"
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
