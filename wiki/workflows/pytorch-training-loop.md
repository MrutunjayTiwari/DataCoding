---
type: workflow
status: active
tags: [pytorch, training-loop, autograd]
updated: 2026-09-04
---

# PyTorch Training Loop

A correct PyTorch loop manages data batches, device placement, model mode, gradient state, loss normalization, and evaluation explicitly.

## Tensor fundamentals

- `unsqueeze` inserts a size-one dimension; `squeeze` removes only dimensions you intend to remove.
- `permute` reorders dimensions; it does not copy values merely to reshape them.
- `torch.cat` joins an existing dimension; `torch.stack` creates a new dimension.
- `torch.from_numpy(array)` shares CPU memory; `torch.tensor(array)` copies.
- Convert model results with `tensor.detach().cpu().numpy()` so autograd and device state are handled explicitly.

## Canonical training phase

```python
model.train()
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(X_batch)
    loss = loss_fn(prediction, y_batch)
    loss.backward()
    optimizer.step()
```

## Canonical evaluation phase

```python
model.eval()
with torch.no_grad():
    for X_batch, y_batch in validation_loader:
        ...
```

`train()`/`eval()` changes Dropout and BatchNorm behavior. `no_grad()` prevents autograd graph construction. They solve different problems and both matter.

## Shape contracts

- Regression: prediction and target often `(batch, 1)` with `MSELoss`.
- Multiclass classification: logits `(batch, classes)`, integer targets `(batch,)`, `CrossEntropyLoss`; do not apply softmax before this loss.
- Binary classification: logits `(batch,)` or `(batch, 1)` aligned with float targets and `BCEWithLogitsLoss`.

## Correct aggregation

If a loss is a batch mean, multiply by batch size before accumulating and divide by the number of examples at the end. Averaging batch means directly biases the result when the final batch is smaller.

## Failure modes

- Gradients accumulate because they were not reset.
- Evaluation accidentally updates BatchNorm statistics.
- Target dtype/shape does not match the loss.
- Calling `.item()` too early detaches a value needed for gradients.
- Calling `.numpy()` directly on a gradient-tracked or non-CPU tensor.
- Validation data is shuffled or augmented like training data.
- Model and tensors reside on different devices.
- Saving the whole model object couples the checkpoint to code layout; prefer `state_dict`.

## Interview drill

Predict `cat`/`stack` and `unsqueeze`/`permute` shapes, then write train and evaluation functions from memory and explain why logits—not probabilities—go into common combined losses.

Executable reference: [PyTorch fundamentals](../../notebooks/04-pytorch/01_pytorch_fundamentals.ipynb).

## Connections

[[wiki/foundations/python-oop|Python OOP]] · [[wiki/workflows/image-classification|Image classification]] · [[wiki/algorithms/logistic-regression|Logistic regression]]
