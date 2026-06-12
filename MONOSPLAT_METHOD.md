# MonoSplat on the CARLA Cross-Sensor (CSE) Benchmark — Method

How MonoSplat (a feed-forward / generalizable 3DGS method) is run on our CARLA
Cross-Sensor benchmark, for both zero-shot inference and CARLA fine-tuning. All
variants are scored with the **same** evaluation script and metric definitions
used for every other entry, so the comparison is apples-to-apples.

---

## 1. Paradigm

MonoSplat predicts pixel-aligned 3D Gaussians from a few posed input views in a
single forward pass — no per-scene optimization. Its geometry comes from a
**frozen monocular foundation** (a DINOv2 encoder + a Depth-Anything DPT head);
trainable multi-view modules fuse the input views and regress the Gaussian
parameters. On each CSE sequence the method reconstructs from the source
(odd-camera) views and renders the target (even-camera) poses, which are unseen
during reconstruction.

## 2. Inference on CSE

For every target pose of a sequence:

1. **Context selection (direction-aware).** Choose the source views whose
   viewing direction aligns with the target (forward-direction dot product above
   a threshold), then take the nearest of them by camera center, as the context
   set. This step is essential on the CARLA surround rig: its cameras share a
   near-common center but face different directions, so naive nearest-center
   selection picks misoriented views and the renders collapse to black.
2. **Encode → Gaussians.** Run the encoder once on the context views to obtain
   pixel-aligned 3D Gaussians.
3. **Render.** Splat the Gaussians at the target pose with the differentiable
   3DGS rasterizer, at the native ground-truth resolution, and save the image
   under the exact target filename.

**Conventions.**
- Poses: COLMAP `(q, t)` are world→camera; we use the OpenCV cam→world
  extrinsics `[R^T | -R^T t]`.
- Intrinsics: resolution-normalized (`fx/W, fy/H, cx/W, cy/H`); the rasterizer
  derives the field of view from these and assumes a centered principal point.
- Depth range (near/far): estimated per sequence, in the scene's coordinate
  units, from the sparse SfM points projected into the source cameras (robust
  percentiles with a small margin). SfM scale is arbitrary, so no dataset-
  specific constants are used; when sparse points are unavailable the scale is
  recovered from the known rig geometry.
- Input resolution: context views are resized so the long side matches the
  network input, preserving aspect ratio and padding to the required multiple.

## 3. CARLA fine-tuning

We fine-tune with the same novel-view photometric objective MonoSplat was
pre-trained with.

- **What is trained.** The monocular foundation (DINOv2 encoder + Depth-Anything
  DPT head) is **kept frozen**; only the multi-view fusion and Gaussian-
  prediction modules are updated. Freezing the foundation preserves the
  generalizable monocular prior and avoids overfitting on the limited CARLA data.
- **Objective.** A purely photometric loss between the rendered target and the
  real target image: pixel MSE plus a small-weight VGG-LPIPS term. No depth or
  3D supervision is used.
- **Per-step sampling.** Each step draws a random source view as the target and
  uses its direction-aligned nearest views as context — the same selection rule
  as inference, so training and test distributions match.
- **Optimization.** Adam with a small learning rate and a short linear warm-up.
  A small learning rate with **early stopping** is important: cross-sensor
  accuracy peaks early and then degrades, because photometric fine-tuning
  overfits same-sensor (source→source) interpolation while the task requires
  source→target cross-sensor extrapolation; larger learning rates rapidly forget
  the pretrained prior.
- **Data.** The CARLA training sequences only; the CSE test sequences are
  strictly excluded from fine-tuning.
- **Checkpoint selection.** The best checkpoint is selected on CSE — the same
  protocol applied to every fine-tuned baseline.

## 4. Evaluation

Every method is scored by a single shared evaluator: PSNR on `[0,1]` RGB, SSIM
with an 11×11 Gaussian window, and LPIPS with a VGG backbone. Renders are
matched to ground-truth target images by filename (the names listed in each
sequence's held-out set), and the render resolution matches the ground truth.

## 5. Code

- `src/scripts/run_monosplat_cse.py` — feed-forward inference / rendering on CSE.
- `src/scripts/finetune_cse.py` — CARLA photometric fine-tuning (foundation
  frozen).

```bash
# Inference / render
python -m src.scripts.run_monosplat_cse \
    --checkpoint <monosplat.ckpt> \
    --multi runs/cse_scenes/<id>:<out>/<id>/renders ...

# CARLA fine-tuning (foundation frozen; early-stopped, best checkpoint on CSE)
python -m src.scripts.finetune_cse \
    --train_glob '<carla_train>/*_dense' \
    --checkpoint <monosplat.ckpt> \
    --out <ft_out> --lr 5e-6 --warmup 100 --save_every 100
```
