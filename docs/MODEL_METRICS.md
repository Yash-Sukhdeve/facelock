# facelock — Recognition Model Metrics (internal reference)

> **Scope — read first.** These are the **recognition model's** score-separability metrics
> (SFace embedder + the deployed max-over-bank matcher), measured on captured embeddings.
> They are **NOT**: an application-level FMR/FNMR under real deployment; a certified biometric;
> validated across multiple subjects, sessions, or demographics; or a spoof-resistance claim.
> **Do not publish as a product/biometric-grade claim or use for marketing.** The facelock
> *application* has not been tested end-to-end as a biometric. This note characterizes the
> recognition backbone only.

## What was measured

Owner captures (415 live probe embeddings, **one subject, one session**) scored against **LFW**
(5,739 identities, one image each) through the exact deployed matcher (max-over-bank, τ=0.363).
Method aligned to ISO/IEC 19795-1; Wilson 95% CIs.

| Model metric | Value | 95% CI | Basis |
|---|---|---|---|
| FMR (per frame) | 0.035 % | 0.010 – 0.127 % | 2 / 5,739 impostor IDs |
| FNMR (per frame) | 0.24 % | 0.043 – 1.35 % | 1 / 415 genuine |
| EER | 0.14 % | 0 – 0.24 % | equal-error crossing |

## What this DOES tell us
The SFace recognition backbone separates this owner from LFW strangers with **sub-percent
equal-error** at the shipped threshold — i.e., the underlying recognizer is strong and the
deployed matcher is scored honestly (no centroid-vs-max-bank understatement).

## What this does NOT tell us (do not infer)
- Real-world application FMR/FNMR under deployment (lighting/pose/aging/distance, multiple users).
- Demographic fairness (LFW is skewed; no breakdown done).
- Any presentation-attack / liveness resistance (that is PAD, measured separately, still pending).

## For the public repo
Publish only the **evaluation harness code** (`facelock/eval/*`, `facelock-eval`) and, if wanted,
this note **verbatim with its scope box** — never a "biometric-grade" product claim. The raw
`eval_report.json` (with provenance + CIs) is the traceable record.

## References (method)
ISO/IEC 19795-1:2021; Wilson (1927) JASA; Huang et al. (2007) LFW Tech Report; SFace/YuNet (OpenCV Zoo).
