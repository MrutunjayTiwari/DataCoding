---
type: workflow
status: active
tags: [computer-vision, image-classification, pytorch]
updated: 2026-09-04
---

# Image Classification

An image-classification system is a data contract plus a training loop: paths/labels → decoded image → tensor transform → logits → metric/prediction.

## Dataset contracts

- Folder layout: `split/class_name/image.ext` works with `ImageFolder`.
- CSV layout: a custom `Dataset` maps filename and label columns to decoded images.
- Unlabeled inference: return `(tensor, filename)` so predictions can be joined back reliably.

Inspect `classes` and `class_to_idx`; never assume alphabetical or portal label ordering without checking.

## Tensor contract

PyTorch convolution expects `NCHW`: `(batch, channels, height, width)`. Images are commonly converted to floating point in `[0, 1]`, then normalized using training-set or pretrained-weight statistics.

Training transforms may include random crop/flip/color changes. Validation/test transforms must be deterministic and aligned with model input size.

## Modeling ladder

1. Majority/random baseline.
2. Linear model on pixels or frozen embeddings.
3. Small CNN from scratch.
4. Pretrained backbone with a replaced classifier head.
5. Fine-tune more layers only when data and compute justify it.

## Evaluation contract

Use validation data for model choices and touch the test set once. Report aggregate loss/accuracy together with a confusion matrix and per-class recall when class behavior matters. Keep predictions keyed by stable filenames or IDs so errors can be inspected and joined back to source records.

## Failure modes

- Train and validation contain near-duplicate images.
- Label mapping changes between splits.
- Augmentation is applied during evaluation.
- Pretrained model receives the wrong normalization or resolution.
- Accuracy hides class imbalance; inspect per-class recall/confusion matrix.
- Downloaded datasets, generated toy images, and weights land inside the vault.

## Interview drill

Implement a `Dataset`, inspect one batch and its min/max/dtype, define a small CNN, write train/eval loops, and emit filename-keyed predictions with a confusion matrix and per-class recall.

Executable reference: [Image classification pattern](../../notebooks/04-pytorch/02_image_classification.ipynb). It uses generated tensors and writes no images into the vault.

## Connections

[[wiki/workflows/pytorch-training-loop|PyTorch loop]] · [[wiki/foundations/numpy|Array shapes]] · [[wiki/workflows/automl|AutoML baselines]]

## Sources and provenance

Consolidates the legacy PyTorch introduction, training-loop drills, CIFAR-10 case study, and portal dataset-pattern notebooks. Large downloads and duplicate optional frameworks were removed from the core path.
