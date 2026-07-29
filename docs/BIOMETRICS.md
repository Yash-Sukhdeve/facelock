# facelock — Biometric System Specification

- **Document type**: Biometrics specification (technical reference for a biometrics researcher)
- **System**: `facelock` — lightweight, RGB-only, single-owner face-unlock daemon for Linux (Prototype "P" profile)
- **Scope**: The face-recognition biometric subsystem (capture → detect → align → embed → match → decide), its operating points, calibration, presentation-attack posture, template protection, and standards mapping.
- **Status**: Describes the *as-built* prototype. Every operating-point number is labelled **measured**, **verified-from-source**, or **design target**.
- **Companion artifacts**: `docs/requirements.md` (REQ-F/REQ-NF/FM/ASM IDs), `docs/design.md` (§3 recognition stack, §11 template store, §13 P↔H boundary), `config/facelock.toml`, `facelock/{detect,embed,matcher,calibrate}.py`.
- **Ground-truth date of model inspection**: 2026-07-29, on the target machine (Ubuntu 24.04, Python 3.11.13, OpenCV 4.12.0, CPU-only).

> **Honesty contract (user rules R1/R6).** This document distinguishes three provenance classes for every claim:
> **[M] Measured** — read directly from the installed artifacts on this machine (ONNX graph, `cv2` output, config, source code).
> **[V] Verified-from-source** — a published number confirmed against an authoritative primary source (paper DOI, OpenCV docs, ISO catalogue). See §11 References.
> **[T] Design target** — an intended operating point from `requirements.md`/`design.md` that has **not** been empirically measured on a real biometric cohort. Targets are **not** results.
> Where a value the reader might expect could not be verified, it is stated as *unverified* rather than guessed.

---

## 0. Provenance summary (what was verified, and how)

### 0.1 Empirically measured from the installed artifacts [M]

Inspection commands (reproducible): `onnx.load()` on each model for graph I/O; `cv2.FaceRecognizerSF.feature()` on a synthetic 112×112 crop for the embedding shape; direct reads of `config/facelock.toml` and the four pipeline source files.

| Fact | Measured value | How measured |
|---|---|---|
| YuNet ONNX file | `face_detection_yunet_2023mar.onnx`, 232,589 bytes (≈0.23 MB) | `stat`; SHA-256 pin file present |
| YuNet SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | bundled `.sha256` |
| YuNet ONNX graph input | tensor `input`, shape `[1, 3, 640, 640]`; IR v6; opset 11; producer `pytorch 1.7` | `onnx.load` |
| YuNet ONNX graph outputs | 12 tensors at strides 8/16/32: `cls_{8,16,32}`, `obj_{8,16,32}`, `bbox_{8,16,32}` (4), `kps_{8,16,32}` (**10 = 5 landmarks × 2**) | `onnx.load` |
| YuNet post-processed row (via `cv2.FaceDetectorYN`) | 15 columns: `bbox(4) + 5 landmarks(10) + score(1)` | `facelock/detect.py` |
| SFace ONNX file | `face_recognition_sface_2021dec.onnx`, 38,696,353 bytes (≈38.7 MB) | `stat`; SHA-256 pin file present |
| SFace SHA-256 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | bundled `.sha256` |
| SFace ONNX graph input | primary input `data`, shape `[1, 3, 112, 112]`; IR v6; opset 11 | `onnx.load` |
| SFace ONNX graph output | tensor `fc1`, shape **`[1, 128]`** | `onnx.load` |
| SFace embedding (empirical) | `cv2.FaceRecognizerSF.feature()` on a 112×112 BGR crop → array shape **(1, 128)**, dtype `float32` | live `cv2` call |
| SFace raw embedding norm | ≈ 5.31 for a random crop (i.e. **not** unit-norm at the model output; `facelock` L2-normalizes downstream) | live `cv2` call + `facelock/embed.py` |
| Operating config | `tau_seed=0.363`, `tau_floor=0.363`, `tau=0.0` (⇒ calibrated), `fmr_target=0.01`, `fnmr_target=0.05`, `probe_frames=5`, `match_votes=3`, `pose_templates=true`, `pose_max=5`, `metric="cosine"`, `confidence_floor=0.90`, `min_face_px=80`, `nms_threshold=0.30`, `liveness.mode="off"`, `pad_model_path=""` | `config/facelock.toml` |

### 0.2 Verified against authoritative published sources [V]

| Fact | Verified value | Source |
|---|---|---|
| SFace paper | Zhong, Deng, Hu, Zhao, Li, Wen, "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition," *IEEE T-IP*, vol. 30, pp. 2587–2598, 2021; DOI 10.1109/TIP.2020.3048632; arXiv:2205.12010 | IEEE Xplore, ADS, PubMed [R1] |
| SFace default cosine op-point | "two faces have the same identity if the **cosine distance ≥ 0.363**, or the normL2 distance ≤ 1.128" | OpenCV DNN-face tutorial [R3] |
| SFace stock accuracy | **99.60 %** on LFW at cosine 0.363 / normL2 1.128 | OpenCV DNN-face tutorial [R3] |
| YuNet paper | Wu, Peng, Yu, "YuNet: A Tiny Millisecond-level Face Detector," *Machine Intelligence Research*, vol. 20, no. 5, pp. 656–665, 2023; DOI 10.1007/s11633-023-1423-y; **75,856** parameters | Springer/MIR [R4] |
| YuNet source project | `ShiqiYu/libfacedetection`; distributed via OpenCV Zoo | libfacedetection, OpenCV Zoo [R5][R6] |
| YuNet WIDER FACE (this ONNX) | AP easy 0.834 / medium 0.824 / hard 0.708 (OpenCV Zoo card) | OpenCV Zoo [R6] |
| Standards years | ISO/IEC 2382-37:2022; ISO/IEC 19795-1:2021; ISO/IEC 30107-3:2023 | ISO catalogue [R7][R8][R9] |

### 0.3 Could NOT be verified — flagged, not guessed

| Item | Status |
|---|---|
| **Runtime latency/throughput** ("~16–26 ms p95 detect+embed", "~13.7 fps active") | **NOT FOUND in the repository.** `requirements.md`/`design.md` state only *targets*: active loop ≥ 5 fps, per-frame `detect+embed+compare` ≤ 200 ms, P95 face→decision ≤ 2.0 s (REQ-NF-01/02). No benchmark harness output, JSON result, or measured fps/ms figure exists in the tree. Any specific measured envelope is therefore **unverified** and is reported here only as a **[T] design target**. (`facelock/cli.py test` can produce a live per-frame p95, but no such measurement is persisted.) |
| SFace packaged-model **backbone** name | The OpenCV Zoo card does not name the backbone. The ONNX graph (≈92 nodes) is dominated by depthwise-separable convolutions (`conv_*_dw`), which is *consistent with* a MobileFaceNet-class lightweight CNN, but the exact backbone is **not asserted by the source** — reported as an observation only. |
| SFace packaged-model **training corpus** | Not stated on the OpenCV Zoo card; the SFace *paper* [R1] evaluates the loss on standard corpora, but the exact training set of *this ONNX file* is **unverified**. |
| System-level FMR/FNMR | The prototype's impostor set is **synthetic** (§6). All FMR/FNMR numbers here are **[T] targets / analytical**, not a measured DET/ROC on a real impostor cohort. |

---

## 1. Pipeline overview

`facelock` performs **1:1 verification** (owner vs. not-owner), single enrolled subject (REQ-F-07, CST-1). The perception pipeline is a strict per-frame chain; a system-level decision is a **k-of-n** temporal fusion over consecutive frames.

```
 ┌────────── per frame (active loop) ─────────────────────────────────────────────┐
 │                                                                                 │
 │  [1] CAPTURE            V4L2 / OpenCV, 640×480 BGR, YUYV, fps_active (config 15) │
 │        │                                                                        │
 │        ▼                                                                        │
 │  [2] DETECT            YuNet ONNX via cv2.FaceDetectorYN                         │
 │        │               → 0..N faces: bbox + det-score + 5 landmarks             │
 │        │               gate: score ≥ 0.90, min(w,h) ≥ 80 px, NMS 0.30           │
 │        ▼                                                                        │
 │  [3] ALIGN            SFace alignCrop(): 5-landmark similarity transform         │
 │        │              → 112×112 aligned BGR crop (ArcFace-style template)        │
 │        ▼                                                                        │
 │  [4] EMBED            SFace ONNX via cv2.FaceRecognizerSF.feature()              │
 │        │              → 128-D float32, then L2-normalized (‖x‖=1)                │
 │        ▼                                                                        │
 │  [5] MATCH (per frame) cosine(x, best-of pose bank) ≥ τ  AND  face_count == 1    │
 │        │               → one boolean "vote"                                     │
 │        ▼                                                                        │
 └────────┼────────────────────────────────────────────────────────────────────────┘
          ▼
   [6] DECIDE (temporal)  sliding window of n=probe_frames votes;
                          grant iff (≥ k=match_votes votes in window)
                          AND (current frame voted yes) AND (exactly 1 face)
                          → is_owner=True → UNLOCK_GRANT (default k-of-n = 3-of-5)
```

Fail-closed rules are enforced at every stage (design §4, SI-P1..P5): a detector error returns `[]`; an embedder error returns `None`; a degenerate/NaN embedding scores −1 (worst); a missing/corrupt template forces `is_owner=False`; `τ` is **never** lowered at runtime (REQ-NF-22). None of these paths can produce an unlock.

**Enrollment (offline)** reuses stages [1]–[4] to collect ≥ 5 accepted samples, builds a centroid + per-pose sample bank, and calibrates the per-owner threshold `τ` (§6). No raw frames are persisted (REQ-NF-13); only embeddings/templates.

---

## 2. Face detector — YuNet

### 2.1 Identity and provenance

| Property | Value | Provenance |
|---|---|---|
| Model file | `face_detection_yunet_2023mar.onnx` (≈0.23 MB) | [M] |
| SHA-256 (pinned) | `8f2383e4…52fa4` | [M] |
| Runtime | OpenCV DNN (`cv2.FaceDetectorYN`), CPU execution provider | [M] design §3 |
| Architecture | **YuNet** — anchor-free, multi-scale (strides 8/16/32) tiny CNN face detector, ~75,856 params | [V][R4] |
| Source project | `ShiqiYu/libfacedetection`; redistributed via OpenCV Zoo | [V][R5][R6] |
| Reference paper | Wu, Peng, Yu, *MIR* 2023, DOI 10.1007/s11633-023-1423-y | [V][R4] |

### 2.2 Input resolution — the subtlety

- The **ONNX graph declares a static input** `input = [1, 3, 640, 640]` [M]. YuNet is **fully convolutional**, so OpenCV's `FaceDetectorYN` **re-sets the input tensor to the frame size at runtime**.
- In `facelock` (`detect.py`), the detector is constructed with a placeholder `(320, 320)` and then `setInputSize((w, h))` is called with the **actual capture frame size = 640×480** before every detect call (`_ensure_input_size`). [M]
- **Net effect**: detection runs on the native **640×480** capture frame (no letterboxing performed by `facelock`); the 640×640 graph value is a declared default, not the operating resolution.

### 2.3 Outputs

Raw ONNX outputs (per scale, before OpenCV post-processing) [M]:

| Head | Stride 8 | Stride 16 | Stride 32 | Meaning |
|---|---|---|---|---|
| `cls_*` | [1,6400,1] | [1,1600,1] | [1,400,1] | classification logit |
| `obj_*` | [1,6400,1] | [1,1600,1] | [1,400,1] | objectness |
| `bbox_*` | [1,6400,4] | [1,1600,4] | [1,400,4] | box regression |
| `kps_*` | [1,6400,10] | [1,1600,10] | [1,400,10] | **5 landmarks × (x,y)** |

`cv2.FaceDetectorYN.detect()` fuses/NMS-filters these into a **15-column per-face row** [M] (`detect.py`):

```
[ x, y, w, h,   x_re,y_re, x_le,y_le, x_nt,y_nt, x_rcm,y_rcm, x_lcm,y_lcm,   score ]
  └── bbox ──┘  └──────────────── 5 landmarks (10 values) ─────────────────┘  └score┘
```

The 5 landmarks, in `facelock`'s order [M] (`detect.py:_LANDMARK_NAMES`): **right eye, left eye, nose tip, right mouth corner, left mouth corner**. These feed both SFace alignment (§3) and the Hardening head-turn liveness geometry (§7).

### 2.4 Confidence and size gating (config)

| Parameter | Config key | Value | Effect | REQ |
|---|---|---|---|---|
| Confidence floor | `detection.confidence_floor` | **0.90** [M] | rows with `score < 0.90` discarded | REQ-F-06, FM-05 |
| Minimum face size | `detection.min_face_px` | **80 px** [M] | `min(w,h) < 80` discarded (rejects distant/tiny faces) | REQ-F-02, FM-05 |
| NMS IoU threshold | `detection.nms_threshold` | **0.30** [M] | passed to `FaceDetectorYN.create` | REQ-F-06 |
| Top-K | (source constant) | 50 [M] | max candidates before NMS | — |

Detections are sorted by descending box area so index 0 is the dominant (nearest) face. A per-frame inference exception returns `[]` and flags the detector unhealthy — an empty detection list is never an unlock (design I-3).

---

## 3. Alignment (5-landmark similarity transform → 112×112)

`facelock` does **not** hand-roll alignment; it delegates to OpenCV's `cv2.FaceRecognizerSF.alignCrop()` [M] (`embed.py`):

```python
aligned = self._rec.alignCrop(bgr, detection.raw_row.reshape(1, -1))  # 15-value YuNet row in
feature = self._rec.feature(aligned)                                   # 112×112 crop → 128-D
```

- `alignCrop` consumes the **full 15-value YuNet row** (bbox + 5 landmarks + score) and computes a **similarity transform** (rotation + uniform scale + translation) that warps the detected 5 points onto SFace's canonical **112×112** reference template. This is the standard **ArcFace/InsightFace-style 5-point alignment** used across the InsightFace lineage; the SFace input tensor `data = [1,3,112,112]` [M] confirms the target crop size.
- Only a similarity (4-DoF) transform is used — enough to normalize in-plane rotation and scale from the two eyes / mouth corners, but **not** a full affine or 3D frontalization. Consequence: large out-of-plane pose (yaw/pitch) is *not* corrected here; `facelock` instead handles off-angle faces at the matching stage via a **multi-pose sub-template bank** (§5.4), not via geometric frontalization.
- Alignment failure returns `None` and is treated as a non-match (fail-closed, FM-05/FM-11). The aligned crop is also exposed via `FaceEmbedder.align()` for the (Hardening) passive-PAD path so PAD does not reach into the recognizer handle.

---

## 4. Face recognizer / embedder — SFace

### 4.1 Identity and provenance

| Property | Value | Provenance |
|---|---|---|
| Model file | `face_recognition_sface_2021dec.onnx` (≈38.7 MB) | [M] |
| SHA-256 (pinned) | `0ba9fbfa…4e79` | [M] |
| Runtime | OpenCV DNN (`cv2.FaceRecognizerSF`), CPU | [M] |
| Input | `data = [1, 3, 112, 112]` (aligned BGR crop) | [M] |
| Output | `fc1 = [1, 128]` float32 (**128-D embedding**) | [M] |
| Loss / method | **Sigmoid-Constrained Hypersphere Loss (SFace)** — intra/inter-class constraints on a hypersphere, controlled by two sigmoid gradient re-scaling functions | [V][R1] |
| Reference paper | Zhong, Deng, Hu, Zhao, Li, Wen, *IEEE T-IP* vol. 30, pp. 2587–2598, 2021; DOI 10.1109/TIP.2020.3048632; arXiv:2205.12010 | [V][R1] |
| Stock accuracy | 99.60 % on LFW @ cosine 0.363 / normL2 1.128 | [V][R3] |
| Backbone | Not named by the OpenCV Zoo card. Observed ONNX is dominated by depthwise-separable convs (`conv_*_dw`), *consistent with* a MobileFaceNet-class lightweight CNN — **observation, not an asserted spec** | [M] + flagged §0.3 |
| Training corpus (of this ONNX) | **Unverified** — not stated by the source | flagged §0.3 |

> **Name-collision caution.** There are two unrelated models called "SFace": (a) Zhong & Deng, *sigmoid-constrained hypersphere loss* (T-IP 2021) — **this is the one packaged here**, and the OpenCV Zoo card links exactly to arXiv:2205.12010; and (b) Boutros et al., *"SFace: Privacy-friendly and Accurate Face Recognition using Synthetic Data"* (IJCB 2022) — a different model. Do not conflate them.

### 4.2 Embedding representation, normalization, metric

- **Dimensionality: 128-D**, verified two independent ways — the ONNX output tensor `fc1 = [1,128]` and a live `cv2.FaceRecognizerSF.feature()` call returning `(1,128)` float32 [M].
- **Normalization**: SFace's `feature()` output is **not** unit-norm (measured raw ‖x‖ ≈ 5.31) [M]. `facelock` **L2-normalizes** every embedding to ‖x‖ = 1 in `embed.py` (`l2_normalize`), rejecting non-finite or zero-norm vectors as `None` (fail-closed). The owner **centroid** is the L2-normalized mean of the L2-normalized samples (`calibrate.centroid_of`) [M].
- **Metric**: **cosine similarity** on the unit-norm embeddings (config `recognition.metric="cosine"`) [M]. Because vectors are unit-norm, cosine = dot product; the code also supports `l2` distance as an alternative. Degenerate inputs score −1.0 (cosine) / +∞ (L2) so a NaN embedding can never match (matcher.py `cosine_similarity`).

---

## 5. Operating point(s)

### 5.1 The published SFace operating point (the seed)

OpenCV's DNN-face tutorial defines the stock SFace decision rule [V][R3]:

> *"two faces have the same identity if the cosine distance is greater than or equal to **0.363**, or the normL2 distance is less than or equal to **1.128**."* (99.60 % LFW)

`facelock` adopts **0.363** as both the calibration **seed** (`tau_seed`) and the hard **floor** (`tau_floor`) — see §6. The published 0.363 is a *population* operating point tuned on LFW; `facelock` treats it as a prior and calibrates per-owner **upward** from it.

### 5.2 The system's configured operating point

| Key (`config/facelock.toml`) | Value [M] | Meaning |
|---|---|---|
| `recognition.metric` | `cosine` | unit-norm cosine similarity |
| `recognition.tau` | `0.0` | `0` ⇒ use the **calibrated** τ from the template (§6); a nonzero value would pin τ |
| `recognition.tau_seed` | `0.363` | calibration prior = SFace published cosine op-point [V] |
| `recognition.tau_floor` | `0.363` | **τ is never calibrated below this** (REQ-NF-22 safety floor) |
| `recognition.fmr_target` | `0.01` | Prototype per-comparison FMR target (1 %) **[T]** |
| `recognition.fnmr_target` | `0.05` | Prototype per-comparison FNMR target (5 %) **[T]** |
| `recognition.probe_frames` (n) | `5` | window size in k-of-n |
| `recognition.match_votes` (k) | `3` | required votes ⇒ **k-of-n = 3-of-5** |
| `recognition.pose_templates` | `true` | score against best-of enrolled pose sub-templates |
| `recognition.pose_max` | `5` | max pose sub-templates in the bank |

### 5.3 Per-frame vs. system-level FMR/FNMR (the k-of-n effect)

Let **p** = per-comparison (single-frame) FMR and **q** = per-comparison FNMR of the SFace+τ classifier. The default fusion requires **≥ k of n** frames to individually accept. Under the (idealized) assumption of **independent, identically-distributed frames**, the number of accepting frames is Binomial, and:

- **System FMR** = P(Binomial(n, p) ≥ k) = Σ_{i=k}^{n} C(n,i) pⁱ(1−p)ⁿ⁻ⁱ ≈ C(n,k)·pᵏ for small p (dominant term).
- **System FNMR** = P(fewer than k genuine accepts) = P(Binomial(n, q) ≥ n−k+1) = Σ_{i=k}^{n} C(n,i) qⁱ(1−q)ⁿ⁻ⁱ (same functional form in q).

For the default **k=3, n=5**, the dominant term is C(5,3) = 10, so both errors fall roughly as **10·(rate)³**:

| Per-frame rate | System FMR = P(Bin(5,p)≥3) | System FNMR = P(Bin(5,q)≥3) |
|---|---|---|
| 0.20 | 5.79 × 10⁻² | 5.79 × 10⁻² |
| 0.15 | 2.66 × 10⁻² | 2.66 × 10⁻² |
| 0.10 | 8.56 × 10⁻³ | 8.56 × 10⁻³ |
| 0.05 | 1.16 × 10⁻³ | 1.16 × 10⁻³ |
| 0.02 | 7.76 × 10⁻⁵ | 7.76 × 10⁻⁵ |
| 0.01 | 9.85 × 10⁻⁶ | 9.85 × 10⁻⁶ |

Reading: an impostor at per-frame FMR = 1 % is driven to a **system FMR ≈ 1.0 × 10⁻⁵** by the 3-of-5 vote — three orders of magnitude tighter. A genuine owner at per-frame FNMR = 5 % is driven to **system FNMR ≈ 1.2 × 10⁻³**.

**Exact facelock rule (refinement).** `facelock` grants only if the **current** frame also passes *and* ≥ k of the window passes (`matcher.verify`: `is_owner = votes_k ≥ k AND face_count==1 AND current_vote`). This is slightly **stricter** than pure ≥k-of-n: the per-decision impostor-accept probability is `p · P(Bin(n−1, p) ≥ k−1)`, e.g. at p=0.01 this is ≈ **5.9 × 10⁻⁶** (vs. 9.85 × 10⁻⁶ for the idealized rule). It also bounds latency: a decision needs at least k frames but at most n.

> **Critical caveat (do not over-read the table).** The i.i.d. assumption is **false** for consecutive video frames of the same subject under fixed illumination — frames are highly **correlated**, so the true variance reduction is far smaller than the binomial predicts. The table is an **upper bound on the benefit**, not a measured operating point. Real system FMR/FNMR must be measured on a temporally-realistic cohort (not available for this prototype, §6/§10).

### 5.4 Multi-pose max-similarity sub-template matching

With `pose_templates=true`, the matcher scores a probe against a **bank** = {centroid} ∪ {up to `pose_max`=5 diverse enrolled sample embeddings}, taking the **maximum** cosine (min L2) over the bank [M] (`matcher._score`, `_build_bank`, `_select_diverse`):

- The pose subset is chosen by **greedy farthest-point (max-min) selection** on the unit-norm sample embeddings, so sub-templates spread across enrolled poses instead of clustering. Deterministic (seeded from row 0) and unit-testable.
- **Effect on errors**: taking a max over P sub-templates *raises* genuine scores for off-angle faces (**lowers FNMR**, enabling easy off-angle auth) but also *raises* impostor scores (a max over P correlated templates ⇒ modest **FMR inflation**). This trade is deliberately bounded by (a) `τ` never being lowered (REQ-NF-22) and (b) the k-of-n vote still gating the decision (FM-02).
- This is `facelock`'s RGB-only, dependency-light substitute for geometric pose frontalization, which is unavailable (no dlib/mediapipe on target).

---

## 6. Calibration methodology (per-owner τ)

Implemented in `facelock/calibrate.py` [M]; realizes design §3.2, REQ-NF-10/22.

### 6.1 Procedure

1. **Genuine distribution (leave-one-out).** For each of the ≥ 5 accepted enrollment samples, score it against the centroid of the *remaining* samples (`_genuine_scores_loo`). This avoids the optimistic bias of scoring a sample against a centroid that includes it.
2. **Impostor distribution.** Score the owner centroid against a bundled impostor **embedding** set (`impostor_embeddings.npz`, embeddings only — **no images**, REQ-NF-13). Calibration requires **≥ 100** impostor embeddings and **≥ 2** genuine samples or it raises.
3. **Threshold from FMR.** Pick the smallest cosine τ whose impostor FMR ≤ `fmr_target` (`_tau_at_fmr_cosine`, using `nextafter` so at most ⌊fmr·m⌋ impostors sit at/above τ).
4. **Enforce the floor.** `τ = max(τ_from_impostor, tau_floor)` [M]. The impostor set can only make τ **tighter**, never weaker than 0.363. If the impostor-derived τ was below the floor, a **warning** is recorded ("floor enforced").
5. **Verify FNMR & report honestly.** Measure achieved FMR/FNMR at the chosen τ; attach **Wilson 95 % confidence intervals** (`wilson_interval`) and `meets_target`. If either target is missed, calibration **warns** and stores the *achieved* operating point with CIs — it **never silently ships a weak τ**, and it **never relaxes τ** to hit FNMR (R1).
6. **Persist** τ + full calibration metadata into the template `.npz` (design §11.2): `{fmr_target, fmr_measured, fnmr_measured, fmr_ci, fnmr_ci, impostor_n, genuine_n, tau_from_impostor, tau_floor, meets_target, metric, calibrated_at, warnings}`.

### 6.2 The τ-floor guarantee (safety property)

`τ ≥ tau_floor = 0.363` is a hard invariant. There is **no code path** — not calibration, not runtime, not re-enroll — that lowers τ below the SFace published cosine op-point. Runtime `τ` is fixed at enrollment and never auto-relaxed to overcome bad light or appearance drift (FM-05/FM-14); the design forbids any threshold-lowering-on-failure branch (REQ-NF-22). This is the recognition-side realization of "fail closed, never fail open."

### 6.3 Honesty: the impostor set is SYNTHETIC

> The prototype's bundled impostor embeddings are a **synthetic / public design set, not a real impostor cohort** matched to the deployment. Therefore the calibrated `fmr_measured` is a **design-target estimate against a stand-in distribution — not a measured DET/ROC** against genuine attackers or a demographically representative population. The Wilson CIs quantify sampling error **on that synthetic set only**; they do **not** capture population, demographic, ageing, or capture-condition mismatch. A defensible FMR/FNMR would require an ISO/IEC 19795-style evaluation on a real cohort (out of scope for the prototype). Treat all §5–§6 error numbers as **[T]**.

---

## 7. Presentation Attack Detection (PAD) / liveness

### 7.1 Prototype: NONE (documented, disclosed)

- Config: `liveness.mode = "off"`, `liveness.pad_model_path = ""` [M]. In the `off`/`P` profile, `LivenessEngine.check()` returns `passed=True` **only** because liveness is disabled — a **documented weakness** (design I-6, REQ-F-17, ASM-04).
- **Consequence (honest):** the prototype is **spoofable by a printed photo or a screen/video replay** of the owner. This matches the disclosed exposure of comparable Linux tools (Howdy) [R2][R3]. It is mitigated **only** by scope, not by detection: a false accept dismisses the tool's own convenience **shield** of an already-logged-in session; it never touches PAM/sudo/login and never removes the OS password path (CST-3, SI-P1..P3). This limitation MUST be disclosed to the user on first run (REQ-F-17).
- No **PAD metric (APCER/BPCER)** can be reported for the prototype because it performs **no PAD**.

### 7.2 Hardening roadmap (not shipped; target only)

Two software-first defences (no IR hardware on target, ASM-06), gated on `security.phase="H"`:

1. **Passive PAD — MiniFASNet.** A small RGB texture/moiré/reflectance CNN (Minivision *Silent-Face-Anti-Spoofing*; MiniFASNetV2 ≈ 0.435 M params, 0.081 G FLOPs) [R10] run on the aligned crop via ONNX Runtime; score-thresholded by `liveness.pad_threshold` (calibrated in H).
2. **Active challenge-response — head-turn.** A randomized yaw challenge (`liveness.turn_yaw_deg=15°`, `challenge_timeout_s=4 s`) computed purely from YuNet's 5 landmarks; defeats a *static* photo. (68-landmark EAR blink is impossible — dlib/mediapipe absent.)

In `full` mode both must pass before a grant. **Target: APCER ≤ 5 % @ BPCER ≤ 5 %** per PAI species, evaluated per **ISO/IEC 30107-3** methodology (REQ-NF-11) [R9]. Metric definitions (ISO/IEC 30107-3):

| Metric | Definition |
|---|---|
| **APCER** | Attack Presentation Classification Error Rate: proportion of attack presentations of a given **PAI species** wrongly classified as bona fide. |
| **BPCER** | Bona-fide Presentation Classification Error Rate: proportion of genuine presentations wrongly classified as attacks. |
| **PAI species** | A class of presentation attack instrument (e.g. print, screen replay, 2D/3D mask). Reported per species; BPCER quoted at fixed APCER points. |

> These are **[T] targets**. No PAD is implemented, calibrated, or measured in the prototype.

---

## 8. Template protection & privacy

| Property | Prototype (P) | Hardening (H) | Provenance |
|---|---|---|---|
| Representation | 128-D `float32` centroid + per-pose sample embeddings + calibration meta | same | [M] design §11.2 |
| On-disk format | NumPy `.npz` (binary) + JSON meta | same bytes, AES-256-GCM encrypted | design §11.2 |
| File perms | `owner.tmpl` **0600**, dir **0700**, owner-only | 0600 + encrypted | design §11.1/11.3 |
| Integrity | (plaintext at 0600) | HMAC-SHA256 tag (`owner.tmpl.sig`) / GCM tag; tamper ⇒ load fails ⇒ **fail closed** | design §11.3, FM-10 |
| Raw frames | **NEVER persisted** — volatile capture buffer only; build-time lint forbids `imwrite` outside disabled debug path (`persist_frames=false`) | same | REQ-NF-13 [M] config |
| Key management | none | key in GNOME Keyring (Secret Service `org.freedesktop.secrets`), referenced by `keyring-ref` | design §11.5 |
| Network exposure | **none** — no sockets except a local `0600` `SO_PEERCRED` control socket (REQ-NF-12) | same | design §10 |
| Erasure | `facelock delete` overwrites-then-unlinks all biometric artefacts, sets `revoked` | same | REQ-F-04 |
| Model binding / revocation | template pins `model_id` = SHA-256 of the SFace model; a model upgrade auto-revokes the template (embeddings not comparable) ⇒ re-enroll | same | design §11.4 |

**Privacy posture.** A face template is biometric personal data. `facelock` minimizes exposure by (1) storing an irreversible-by-design 128-D embedding rather than images (no raw frames on disk, unlike Howdy's snapshot weakness [R3]), (2) owner-only file perms, (3) integrity/encryption in H, and (4) zero network egress. **Honest caveat:** a 128-D embedding is **not** a formally protected biometric template in the ISO/IEC 24745 sense — it is not cancellable/renewable and offers **no template-protection unlinkability guarantee**; 0600 perms + optional keyring encryption are access control, not a biometric-template-protection scheme. Embedding inversion (reconstructing a face image from an embedding) is a known research threat and is not defended against here.

---

## 9. Standards mapping & vocabulary

Terms in this document follow the harmonized biometric vocabulary and are used with their standardized meaning.

| Standard (verified year) | Role in this system | Where it appears |
|---|---|---|
| **ISO/IEC 2382-37:2022** — Vocabulary — Part 37: Biometrics [R7] | Harmonized terms: biometric *reference/template*, *probe*, *comparison*, *verification (1:1)*, *enrolment*, *PAI*. | Throughout |
| **ISO/IEC 19795-1:2021** — Biometric performance testing and reporting — Part 1 [R8] | Performance vocabulary & framework: **FMR, FNMR, FTA** (failure-to-acquire), **FTE** (failure-to-enrol), DET/ROC methodology. | §5, §6, §10 |
| **ISO/IEC 30107-3:2023** — Presentation attack detection — Part 3: Testing and reporting [R9] | PAD evaluation metrics **APCER/BPCER/ACER** and PAI-species reporting. | §7 |
| **NIST FRVT / FRTE-FATE** (1:1 verification benchmarking) [R11] | Context reference establishing FMR/FNMR as the standard 1:1 verification metrics; server-grade algorithms reach very low error (per NIST, ≈ FNMR 0.1–1 % at FMR 1e-5) — `facelock`'s lightweight local targets are deliberately looser and framed against this. | §10 |

Mapping of `facelock` quantities to ISO/IEC 19795 vocabulary:

| facelock quantity | ISO/IEC 19795-1 term |
|---|---|
| cosine `< τ` for the owner across k-of-n | contributes to **FNMR** (false non-match) |
| impostor cosine `≥ τ` across k-of-n | contributes to **FMR** (false match) |
| detector below `confidence_floor`/`min_face_px`; no usable face | **FTA** (failure-to-acquire) — presentation yields no comparison |
| enrollment quality gate rejects all samples / calibration cannot proceed | **FTE** (failure-to-enrol) |

---

## 10. Honest limitations & measured operating envelope

| # | Limitation | Detail | Class |
|---|---|---|---|
| L1 | **RGB-only sensor** | No IR/depth (ASM-06). No structured-light or 3D anti-spoof possible; PAD is software-only (§7). | fact [M] design §1 |
| L2 | **No PAD in the prototype** | `liveness.mode=off`; photo/replay-spoofable, disclosed (REQ-F-17). | fact [M] |
| L3 | **Single enrolled owner** | 1:1 verification only (CST-1). No identification (1:N), no multi-user. | fact [M] |
| L4 | **Synthetic calibration** | Impostor set is synthetic ⇒ FMR/FNMR are **design targets**, not measured DET/ROC on a real cohort. | [T] §6.3 |
| L5 | **Correlated frames** | The k-of-n binomial gain (§5.3) assumes i.i.d. frames; real frames are correlated ⇒ true error reduction is smaller than tabulated. | analysis §5.3 |
| L6 | **Lightweight model** | SFace (99.60 % LFW) is far weaker than server-grade ArcFace/NIST algorithms; accepted for the pilot (ASM-05), disclosed. | [V] design §3 |
| L7 | **No geometric frontalization** | Similarity-transform alignment only; large yaw/pitch handled by the pose bank (§5.4), which can inflate FMR. | [M] §3/§5.4 |
| L8 | **Embedding ≠ protected template** | 128-D embedding is not cancellable/unlinkable (ISO/IEC 24745); inversion threat undefended. | §8 |
| L9 | **CPU-only real-time constraint** | GPU unusable (NVML mismatch, ASM-07); all inference on CPU with a bounded thread pool. | fact [M] design §1 |

### 10.1 Operating envelope — targets, and an honesty flag

The recognition pipeline is dimensioned to real-time on CPU, but **no measured latency/throughput figures exist in the repository.** The following are **[T] design targets** only:

| Quantity | Target (from requirements) | Class |
|---|---|---|
| Active perception loop | **≥ 5 fps** on the target CPU (AC-NF-01) | [T] REQ-NF-01 |
| Per-frame `detect + embed + compare` | **≤ 200 ms** on CPU (AC-NF-02) | [T] REQ-NF-02 |
| Face→unlock decision, P95 | **≤ 2.0 s** (AC-NF-02) | [T] REQ-NF-02 |
| Configured active capture rate | `camera.fps_active = 15` (config; the Brio does 640×480 YUYV @ 30) | [M] config |
| Idle capture rate | `camera.fps_idle = 4` | [M] config |

> **Flag (R1).** Any specific measured envelope such as "≈16–26 ms detect+embed p95" or "≈13.7 fps active" is **NOT present anywhere in this repository** and could not be verified. `facelock/cli.py test` computes a live per-frame p95 against the 200 ms budget, but no such measurement is stored. Until the benchmark harness (AC-NF-01/02) is run and its output committed, the operating envelope stands as a **target, not a result.** Do not cite fps/latency numbers as measured.

---

## 11. References

Verified citations (primary sources checked 2026-07-29). Per rule R6, model/spec facts trace to authoritative sources; no citation was hand-fabricated.

- **[R1]** Y. Zhong, W. Deng, J. Hu, D. Zhao, X. Li, D. Wen, "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition," *IEEE Transactions on Image Processing*, vol. 30, pp. 2587–2598, 2021. DOI: 10.1109/TIP.2020.3048632. arXiv:2205.12010. (Verified via IEEE Xplore doc 9318547, NASA ADS `2021ITIP...30.2587Z`, PubMed 33417553.)
- **[R2]** Howdy — facial authentication for Linux. https://github.com/boltgolt/howdy ; ArchWiki security notes: https://wiki.archlinux.org/title/Howdy
- **[R3]** "Howdy, Friend" — *Linux Magazine*, issue 256 (2022): documented photo-spoof exposure and on-disk snapshot weakness; "not more secure than a password." https://www.linux-magazine.com/Issues/2022/256/Howdy
- **[R4]** W. Wu, H. Peng, S. Yu, "YuNet: A Tiny Millisecond-level Face Detector," *Machine Intelligence Research*, vol. 20, no. 5, pp. 656–665, 2023. DOI: 10.1007/s11633-023-1423-y.
- **[R5]** S. Yu et al., *libfacedetection* (source project for YuNet). https://github.com/ShiqiYu/libfacedetection
- **[R6]** OpenCV Zoo — model cards for `face_detection_yunet` and `face_recognition_sface`. https://github.com/opencv/opencv_zoo
- **[R7]** ISO/IEC 2382-37:2022, *Information technology — Vocabulary — Part 37: Biometrics*. https://www.iso.org/standard/73514.html
- **[R8]** ISO/IEC 19795-1:2021, *Information technology — Biometric performance testing and reporting — Part 1: Principles and framework*. https://www.iso.org/standard/73515.html
- **[R9]** ISO/IEC 30107-3:2023, *Information technology — Biometric presentation attack detection — Part 3: Testing and reporting*. https://www.iso.org/standard/79520.html
- **[R3-cv]** OpenCV, "DNN-based Face Detection And Recognition" tutorial (SFace cosine 0.363 / normL2 1.128, 99.60 % LFW). https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
- **[R10]** Minivision, *Silent-Face-Anti-Spoofing* (MiniFASNet; MiniFASNetV2 ≈ 0.435 M params). https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- **[R11]** NIST Face Recognition Vendor Test (FRVT) / FRTE-FATE — 1:1 verification benchmarking program and FMR/FNMR methodology. https://pages.nist.gov/frvt/ ; NIST IR 8429 (demographics): https://pages.nist.gov/frvt/reports/demographics/nistir_8429.pdf

---

*Provenance legend:* **[M]** measured on the installed artifacts (2026-07-29) · **[V]** verified against a primary published source (§11) · **[T]** design target, not measured. Numbers labelled [T] are engineering intent and MUST NOT be reported as biometric results.
