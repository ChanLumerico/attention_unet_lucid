# Attention U-Net on Kvasir-SEG with `💎lucid`

This repository is a focused experiment repo built to stress-test my personal deep learning framework, [`lucid`](https://github.com/ChanLumerico/lucid), under a realistic medical image segmentation workload. Rather than re-implementing a model from scratch here, the core architecture is the already-implemented `AttentionUNet2d` from `lucid`, and this repo acts as the experiment harness around it: dataset preprocessing, EDA, augmentation validation, training configuration, evaluation, and qualitative diagnosis.

From a portfolio perspective, this project is meant to demonstrate three things:

1. `lucid` can support a non-trivial dense prediction pipeline end-to-end.
2. The framework is usable for reproducible experimentation beyond toy classification examples.
3. I can structure a compact but research-oriented experiment repo with clear methodology, diagnostics, and artifact organization.

## Abstract

I trained an **Attention U-Net** style 2D segmentation model on the **Kvasir-SEG** polyp segmentation dataset to evaluate whether `lucid` is stable and expressive enough for a practical encoder-decoder medical vision task. The full pipeline includes deterministic data preprocessing, synchronized geometric augmentation for image-mask pairs, YAML-driven experiment configuration, checkpointed training, and post-hoc prediction diagnostics.

The final model has **21.7M parameters** and was trained on **800 training** images with evaluation on **200 validation** images. With the repository's configured inference threshold of **0.20**, the saved `best` checkpoint reaches **0.412 mean Dice** and **0.288 mean IoU** on the validation split. A threshold sweep on the saved logits indicates that the same checkpoint reaches its best validation operating point near **0.60**, where performance improves to **0.455 mean Dice** and **0.325 mean IoU**.

This suggests that the learned model is reasonably functional but calibrated toward over-segmentation under the current default threshold, which is a useful outcome for testing framework behavior, loss design, and post-processing sensitivity.

## Project Scope

The repository is intentionally split into two layers:

- `lucid`: the underlying DL framework and the implementation source of `AttentionUNet2d`.
- This repo: the experiment layer that validates `lucid` on a real segmentation benchmark.

The main assets are:

- [`source/preprocessing.py`](source/preprocessing.py): dataset download, extraction, resizing, binary mask conversion, and cached split generation.
- [`source/data.py`](source/data.py): `KvasirSegDataset`, synchronized augmentation, and dataloader factories.
- [`source/model.py`](source/model.py): YAML-to-`AttentionUNetConfig` serialization and model construction.
- [`source/train.py`](source/train.py): experiment config parsing, training loop, checkpointing, metrics, plotting, and prediction diagnostics.
- [`source/eda.py`](source/eda.py): dataset summary and visualization routines.
- [`main.ipynb`](main.ipynb): notebook that ties the entire experiment together.
- `out/`: generated figures used for analysis and presentation.

## Research Question

The working question behind this experiment was:

> Can a custom framework (`lucid`) train and evaluate an attention-gated U-Net reliably on a clinically relevant binary segmentation task, while exposing enough ergonomics for reproducible experimentation and useful failure analysis?

This is not positioned as a state-of-the-art claim. It is a framework validation study using a standard medical segmentation architecture and a well-known public dataset.

## Dataset

### Benchmark

- **Dataset**: Kvasir-SEG
- **Task**: binary semantic segmentation of colorectal polyps
- **Original dataset size**: 1,000 RGB endoscopic images with pixel-wise masks
- **Local split protocol in this repo**: random 80/20 split with fixed seed `42`
- **Resulting split sizes**: 800 train / 200 validation
- **Input resolution used for training**: `256 x 256`

### Preprocessing

The preprocessing pipeline in [`source/preprocessing.py`](source/preprocessing.py) does the following:

- downloads the official Kvasir-SEG zip archive,
- extracts image and mask files,
- resizes RGB images with bilinear interpolation,
- resizes masks with nearest-neighbor interpolation,
- converts masks to binary `{0, 1}`,
- caches arrays into compressed `.npz` files for fast repeated experiments.

The cached tensors are stored as:

- images: `(N, 3, 256, 256)`, `float32`, range `[0, 1]`
- masks: `(N, 1, 256, 256)`, `float32`, binary

### Dataset Statistics

The following statistics were recomputed from the cached split used by this repository:

<div align="center">

| Split | N | Mean Pixel Value | Pixel Std | Mean Foreground % | Std Foreground % | Min % | Max % | Median % | P90 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 800 | 0.3705 | 0.2819 | 15.35 | 12.71 | 0.48 | 81.22 | 11.70 | 32.20 |
| Val | 200 | 0.3756 | 0.2832 | 15.58 | 14.09 | 1.08 | 76.44 | 10.74 | 31.84 |

</div>

Interpretation:

- The train/validation splits are well matched in low-level image statistics.
- Foreground occupancy is highly variable, which is expected in lesion segmentation.
- Median polyp coverage is only around `11%`, so the task is meaningfully imbalanced in pixel space.

## Data Pipeline and Augmentation

The training loader applies **joint spatial augmentation** to image-mask pairs using a synchronized random seed so that masks remain geometrically aligned with the image:

- random horizontal flip with `p=0.5`
- random vertical flip with `p=0.2`
- random rotation within `±15°`

After spatial transforms, masks are re-binarized at `0.5` to remove interpolation artifacts. Images are then normalized with ImageNet statistics:

- mean: `(0.485, 0.456, 0.406)`
- std: `(0.229, 0.224, 0.225)`

This design matters because it tests whether `lucid` can support practical segmentation augmentation where image and mask transforms must remain perfectly synchronized.

## Model

### Backbone Under Test

The model instantiated in this repo is `lucid.models.AttentionUNet2d`, configured through YAML and built in [`source/model.py`](source/model.py).

### Architecture Summary

- Input channels: `3`
- Output channels: `1`
- Encoder channels: `[64, 128, 256, 512]`
- Bottleneck channels: `1024`
- Decoder stages: auto-derived from encoder
- Normalization: batch norm
- Activation: ReLU
- Downsampling: max pooling
- Upsampling: bilinear interpolation
- Deep supervision: enabled
- Attention gating: enabled
- Auto-resolved skip gating pattern: `(True, True, False)`
- Parameter count: **21,700,933**

### Why Attention U-Net?

Attention U-Net is a good framework validation target because it is more demanding than a plain CNN classifier:

- it requires multi-scale encoder-decoder wiring,
- uses skip connections with learned gating,
- emits dense spatial predictions,
- optionally returns auxiliary outputs for deep supervision,
- and is sensitive to data augmentation, interpolation, and threshold calibration.

If a custom framework handles this cleanly, that is a stronger signal of engineering maturity than a minimal feed-forward benchmark.

## Training Protocol

The experiment configuration is stored in [`config/attention_unet.yaml`](config/attention_unet.yaml).

### Optimization Setup

<div align="center">

| Component | Setting |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Weight decay | `5e-5` |
| Scheduler | ReduceLROnPlateau |
| Scheduler monitor | validation loss |
| Scheduler factor | `0.5` |
| Scheduler patience | `6` |
| Minimum LR | `1e-6` |
| Batch size | `8` |
| Epoch budget | `120` |
| Device | `gpu` |
| Seed | `42` |

</div>

### Loss Function

The training objective combines binary cross-entropy and soft Dice loss, with optional auxiliary supervision from deep supervision heads:

\[
\mathcal{L} = 0.25 \cdot \mathcal{L}_{BCE} + 0.75 \cdot \mathcal{L}_{Dice} + 0.10 \cdot \mathcal{L}_{aux}
\]

This weighting intentionally biases training toward overlap quality rather than pure pixel-wise calibration.

### Checkpointing and Artifacts

Training writes artifacts under `checkpoints/<config_name>/` and visualization outputs under `out/<config_name>/`. The training script supports:

- saving both `latest` and `best` checkpoints,
- automatic resume from a compatible latest checkpoint,
- configuration persistence for reproducibility,
- training-curve generation after each epoch.

## Evaluation Protocol

Because this repo only contains a train/validation split, the validation set functions as a held-out evaluation split for this project. Metrics are computed at the pixel-mask level after thresholding sigmoid probabilities.

### Metrics

- **Dice coefficient**: primary overlap metric
- **IoU**: stricter overlap metric
- **Foreground Dice**: equivalent here because all validation masks contain foreground
- **Prediction coverage ratio**: mean predicted positive area divided by image area

For reproducibility, the values below were recomputed from the saved `best` checkpoint included in this repository using Python `3.14` and the local `lucid` package.

## Quantitative Results

### Main Validation Results

| Checkpoint | Threshold | Val Loss | Mean Dice | Median Dice | Dice Std | Mean IoU | Mean Predicted FG % | Mean Target FG % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best` | 0.20 | 0.6848 | 0.4117 | 0.4034 | 0.2330 | 0.2884 | 45.55 | 15.58 |
| `best` | 0.50 | 0.6848 | 0.4488 | 0.4512 | 0.2313 | 0.3194 | 34.00 | 15.58 |
| `best` | 0.60 | 0.6848 | **0.4549** | **0.4674** | 0.2305 | **0.3248** | 31.46 | 15.58 |

### Threshold Sweep

The model is quite sensitive to threshold choice:

| Threshold | Mean Dice | Mean IoU | Mean Predicted FG % |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.3586 | 0.2440 | 59.73 |
| 0.20 | 0.4117 | 0.2884 | 45.55 |
| 0.30 | 0.4319 | 0.3054 | 40.03 |
| 0.40 | 0.4417 | 0.3134 | 36.70 |
| 0.50 | 0.4488 | 0.3194 | 34.00 |
| 0.60 | **0.4549** | **0.3248** | 31.46 |

### Interpretation of the Results

The main technical takeaway is not just the absolute Dice value, but the behavior of the system:

- The model clearly learns meaningful lesion localization and shape structure.
- The default threshold in the config (`0.20`) is aggressively recall-oriented and produces systematic over-segmentation.
- Performance increases monotonically across the tested threshold range up to `0.60`, indicating that calibration/post-processing is still an open improvement lever.
- Even at `0.60`, the model predicts substantially larger foreground area than the ground truth on average, so the network still tends to over-cover the lesion region.

In other words, the current training recipe is **functional but not yet calibrated**. For a framework validation repo, that is a useful finding because it surfaces model-behavior questions without exposing any instability in the data or training code.

## Qualitative Analysis

### Sample and Distribution Inspection

Random sample overlays, foreground-area distributions, and spatial heatmaps indicate that the split is visually coherent and that lesions are not restricted to a single trivial location in frame.

<div align="center">
    <img src="out/random_sample_viz.png" width="60%">
    <br>
    Figure 1. Train Samples
    <br>
</div>

<br>

<div align="center">
    <img src="out/polyp_coverage_dist.png" width="80%">
    <br>
    Figure 2. Poly-p Coverage Statistics
    <br>
</div>

<br>

<div align="center">
    <img src="out/spatial_mask_heatmap.png" width="80%">
    <br>
    Figure 3. Average Poly-p Location Across Samples
    <br>
</div>

### Augmentation Sanity Check

The augmentation preview confirms that image-mask synchronization is preserved under flipping and rotation, which is a non-negotiable requirement for segmentation training.

<div align="center">
    <img src="out/aug_preview.png" width="70%">
    <br>
    Figure 4. Augmentation Preview (Train Split)
    <br>
</div>

### Training Dynamics

The final curves indicate that the experiment reaches a stable training regime with sensible separation between optimization progress and validation behavior. This part of the repo is especially important for framework testing because silent autograd, tensor-shape, or checkpoint issues often show up first in curve pathologies.

<div align="center">
    <img src="out/final_curve.png" width="90%">
    <br>
    Figure 5. Final Train Plots
    <br>
</div>

### Prediction Diagnostics

The diagnostic plots and overlay grids show that the model usually captures the general lesion region but often predicts masks that are spatially too large or too diffuse. That is consistent with the threshold sweep and foreground-ratio statistics above.

<div align="center">
    <img src="out/pred_diagnosis.png" width="90%">
    <br>
    Figure 6. Prediction Diagnostics
    <br>
</div>

<br>

<div align="center">
    <img src="out/pred_overlay_grid.png" width="70%">
    <br>
    Figure 7. Test-Split Prediction Overlays (Red: GT, Green: Prediction)
    <br>
</div>

## What This Repo Demonstrates About `lucid`

This experiment was useful as a framework-level validation because it exercised:

- YAML-configurable model construction through dataclass-backed configs,
- multi-output models with deep supervision,
- optimizer and scheduler abstractions,
- dataset and dataloader APIs,
- transform composition for segmentation,
- GPU training and inference,
- checkpoint serialization,
- post-training diagnostic tooling.

The fact that this entire pipeline can be run with a custom framework on a real dense prediction task is the main engineering result of the project.

## Reproducibility

### Run the Full Pipeline

1. Ensure Python `3.14` is available.
2. Make sure the proper `lucid` version is downloaded.

```bash
pip install lucid-dl==2.15.7
```

3. Build the dataset cache:

```bash
python3.14 source/preprocessing.py
```

4. Run the notebook:

```bash
jupyter notebook main.ipynb
```

Or train directly from the YAML config:

```bash
python3.14 source/train.py --config config/attention_unet.yaml
```

### Load Weights Within Lucid

You can directly load the trained model weights in `checkpoints/attention_unet/best/model.safetensors` via Lucid.

```python
import lucid
from lucid.models import AttentionUNet2d

# current repo
from source.model import load_config

config = load_config("config/attention_unet.yaml")

model = AttentionUNet2d(config).to("gpu")  # Apple MLX

model.load_state_dict(
    lucid.load("checkpoints/attention_unet/best/model.safetensors")
)
model = model.compile()  # optional
model.eval()
```

### Repo Outputs

- Cached arrays are written under `cache/`
- Model checkpoints are written under `checkpoints/`
- Figures are written under `out/`

## Limitations and Next Steps

This is a strong framework validation repo, but not yet a complete medical segmentation study. The main limitations are:

- no separate test split or cross-validation,
- only one architecture setting is evaluated,
- threshold calibration is not optimized during training,
- no comparison against a plain U-Net baseline,
- no external benchmark against PyTorch or MONAI for parity checking.

The most meaningful next steps would be:

1. add a plain U-Net baseline inside the same training harness,
2. tune threshold selection on a calibration subset rather than fixing `0.20`,
3. test higher input resolutions such as `352` or `512`,
4. compare `lucid` runtime and numerical behavior against a PyTorch reference implementation,
5. extend evaluation with boundary-sensitive metrics and lesion-wise error analysis.

## Conclusion

As an experiment repo, this project achieved its goal. It shows that `lucid` can support a non-trivial medical image segmentation workflow end-to-end, including preprocessing, augmentation, training, checkpointing, and diagnostics. The resulting Attention U-Net model is not yet fully optimized, but it is clearly learning the task and exposes interpretable calibration behavior.

---

### References

1. Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the Pancreas." arXiv, 2018, https://doi.org/10.48550/arXiv.1804.03999.

2. Jha, Debesh, et al. "Kvasir-SEG: A Segmented Polyp Dataset." MultiMedia Modeling: 26th International Conference, MMM 2020, Daejeon, South Korea, January 5–8, 2020, Proceedings, Part II, edited by Jakub Lokoč et al., Springer, 2020, pp. 451-62, doi.org.
