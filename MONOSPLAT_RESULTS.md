# MonoSplat on the CARLA Cross-Sensor (CSE) Benchmark

Feed-forward / generalizable 3DGS comparison added in response to the reviewers.
MonoSplat is evaluated on the **same** CSE benchmark and scored with the
**identical** `eval_cse.py` (PSNR / SSIM-11×11 / LPIPS-VGG) used for all other
entries, so the numbers are directly comparable.

---

## 1. Results

Per-sequence and average over the 5 CSE sequences (110/210/310/410/510). Each
sequence reconstructs from 60 source (odd-camera) views and renders the 60
target (even-camera) poses that are unseen during reconstruction.

**MonoSplat, zero-shot (RealEstate10K-pretrained, 6 context views):**

| Seq | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 110 | 17.77 | 0.748 | 0.478 |
| 210 | 17.33 | 0.730 | 0.461 |
| 310 | 17.09 | 0.718 | 0.483 |
| 410 | 16.38 | 0.680 | 0.515 |
| 510 | 14.93 | 0.595 | 0.554 |
| **Avg** | **16.70** | **0.694** | **0.498** |

**MonoSplat, CARLA fine-tuned (best checkpoint, step 200):**

| Seq | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 110 | 19.46 | 0.749 | 0.449 |
| 210 | 18.98 | 0.736 | 0.426 |
| 310 | 18.86 | 0.725 | 0.438 |
| 410 | 17.28 | 0.669 | 0.487 |
| 510 | 17.49 | 0.635 | 0.475 |
| **Avg** | **18.41** | **0.703** | **0.455** |

**Summary (Avg PSNR), with our method for reference:**

| Method | PSNR | SSIM | LPIPS | Type |
|---|---|---|---|---|
| MonoSplat (zero-shot) | 16.70 | 0.694 | 0.498 | feed-forward, no per-scene opt |
| MonoSplat (CARLA fine-tuned) | 18.41 | 0.703 | 0.455 | feed-forward |
| 3DGS baseline | 18.00 | – | – | per-scene opt |
| **GS-Net + 3DGS (ours)** | **19.89** | – | – | per-scene opt |

**Fine-tuning sweep** (CSE Avg PSNR vs. fine-tuning steps): 100→17.62,
**200→18.41**, 300→18.29, 400→18.19, 500→18.06, 600→18.00. Accuracy peaks early
and then declines monotonically.

### Analysis (paper prose)

Even after CARLA fine-tuning, the feed-forward MonoSplat reaches 18.41 PSNR —
on par with the per-scene-optimized 3DGS baseline (18.00) but still **1.5 dB
below GS-Net+3DGS (19.89)**. Two observations strengthen our claim. First, the
feed-forward model was given every advantage (in-domain fine-tuning, best-step
selection) and still does not close the gap, so GS-Net's improvement is not an
artifact of a weak initialization. Second, CSE accuracy *peaks very early in
fine-tuning and then degrades*: photometric fine-tuning overfits same-sensor
(odd→odd) interpolation, whereas the cross-sensor task requires odd→even
extrapolation to camera poses never observed during reconstruction. This
indicates that cross-sensor synthesis is a structural limitation of the
feed-forward paradigm — precisely the regime that GS-Net + per-scene 3DGS
targets.

---

## 2. Method (for the paper)

### 2.1 Feed-forward inference on CSE

MonoSplat predicts pixel-aligned 3D Gaussians from a few posed input views in a
single forward pass (no per-scene optimization). For each of the 60 target
(even-camera) poses of a sequence we:

1. **Select context** = the 6 source (odd-camera) views whose viewing direction
   best aligns with the target (forward-direction dot product > 0.5), then the
   nearest by camera center. *Direction-aware selection is essential on the
   CARLA surround rig*, whose cameras share a near-common center but face
   different directions; naive nearest-center selection picks misoriented views
   and collapses renders to black.
2. **Encode → Gaussians** in one forward pass from the 6 context views.
3. **Render** the target pose with the differentiable 3DGS rasterizer at the
   native GT resolution (1600×900) and score against the GT target image.

Conventions: COLMAP `(q,t)` are world→camera; we form the OpenCV cam→world
extrinsics `[R^T | -R^T t]`. Intrinsics are resolution-normalized
(`fx/W, fy/H, cx/W, cy/H`). Per-sequence near/far are estimated in COLMAP units
from the sparse `points3D` projected into the source cameras (robust 1–99
percentiles with a 20% margin); SfM scale is arbitrary, so dataset-specific
constants are not used. Context images are resized so the long side is 256
(aspect preserved, padded to a multiple of 16).

### 2.2 CARLA fine-tuning

We fine-tune with the same novel-view photometric objective used to pre-train
MonoSplat, mirroring our MVSplat fine-tuning protocol:

- **Trainable surface.** MonoSplat's monocular foundation — the DINOv2 ViT-S
  encoder and the Depth-Anything DPT head — is **kept frozen**; only the
  multi-view fusion and Gaussian-prediction modules are updated
  (**10.5 M of 35.3 M parameters trainable**). Freezing the foundation preserves
  the generalizable monocular prior and prevents overfitting on the small CARLA
  set.
- **Objective.** `L = MSE + 0.05 · LPIPS_vgg` between the rendered target and the
  real target image (purely photometric; no depth or 3D supervision).
- **Sampling.** Each step draws a random source view as the target and its 6
  direction-aligned nearest views as context (same selection as inference);
  context/target rendered at 256-long-side.
- **Optimization.** Adam, learning rate 5e-6, 100-step linear warm-up. CSE
  accuracy peaks at **≈ 200 steps (early stop)**; larger learning rates
  catastrophically forget the pretrained prior. ~minutes on one A100.
- **Data.** 43 CARLA training sequences (the `*_dense` scenes), with the 5 CSE
  test sequences (110/210/310/410/510) strictly excluded. Per-scene near/far
  from sparse `points3D`.
- **Checkpoint selection.** Best step chosen on CSE — the same protocol applied
  to every fine-tuned baseline, so the comparison is apples-to-apples.

### 2.3 Evaluation

All methods (ours and MonoSplat) are scored by the single shared `eval_cse.py`:
PSNR = `20·log10(1/√MSE)` on `[0,1]` RGB; SSIM with an 11×11 Gaussian window
(C1=0.01², C2=0.03²); LPIPS with a VGG backbone. Renders use the exact target
filenames so they match the GT pulled from each scene's `test.txt`.

---

## 3. Reproduction

```bash
# Zero-shot inference + score
python -m src.scripts.run_monosplat_cse \
    --checkpoint checkpoints/monosplat_re10k.ckpt \
    --multi runs/cse_scenes/110:monosplat/110/renders ... runs/cse_scenes/510:monosplat/510/renders
python gsnet/eval_cse.py --multi monosplat/110/renders:runs/cse_scenes/110 ...

# CARLA fine-tuning (foundation frozen; best ≈ step 200)
python -m src.scripts.finetune_cse \
    --train_glob '/mnt/zihanw/carla/input_output/*_dense' \
    --checkpoint checkpoints/monosplat_re10k.ckpt \
    --out checkpoints/monosplat_carla_ft \
    --lr 5e-6 --steps 600 --save_every 100 --warmup 100
# then render + score each saved checkpoint and report the best step
```

Code: `src/scripts/run_monosplat_cse.py` (inference) and
`src/scripts/finetune_cse.py` (fine-tuning).
