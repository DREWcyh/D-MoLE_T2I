# D-MoLE v3 Core Logic

## Scope

This document summarizes the core logic implemented by the v3 pipeline:

- `train_dmole_v3.py`
- `infer_dmole_v3.py`
- `feature_extractor_v3.py`
- `router_v3.py`
- `zcp_allocator.py`

## Overview

Version 3 replaces learned routing with a tuning-free residual prototype router.
Its core principle is:

- use the pretrained text space directly,
- subtract a generic base prompt feature,
- L2-normalize the residual,
- store one prototype per task,
- route by nearest prototype under cosine distance.

This removes the extra optimization burden introduced by v1 autoencoders and
v2 text-to-vision mappers.

## Training Flow

### 1. Load the base PixArt components

As in v1 and v2, the system uses:

- `T5Tokenizer`
- `T5EncoderModel`
- `AutoencoderKL`
- `Transformer2DModel`

Only the current LoRA adapter is updated.

### 2. Allocate a new expert

The expert construction path is unchanged:

- ZCP selects high-saliency transformer blocks,
- the new adapter is attached only to those blocks,
- old experts remain frozen.

### 3. Extract projected text features

`feature_extractor_v3.py` only does prompt-level text projection:

1. max-pool T5 hidden states over tokens,
2. project from 4096 to 512 with a deterministic orthogonal matrix,
3. return the raw projected text feature in `float32`.

Residual subtraction and normalization are intentionally not done here. They are
handled inside the router so the same rule is used during both task
registration and inference.

### 4. Register the global base feature

At startup, the trainer encodes a generic prompt:

`A photo of a item`

This projected feature becomes `base_feat` inside the router.

### 5. Mentor lookup for warm-start inheritance

For a new task, the trainer extracts the task prompt feature and queries the
router before training:

- if the nearest stored prototype is close enough, the new adapter inherits
  the matched old expert's weights,
- otherwise the new adapter starts from a fresh initialization.

### 6. Register one prototype per task

`ResidualPrototypeRouter` stores:

- `base_feat`: one global base prompt feature,
- `task_prototypes`: one frozen prototype per task.

Task registration does not include any optimizer or loss. The router:

1. converts the task prompt feature into a residual feature,
2. L2-normalizes it,
3. saves it as the task prototype.

## Router Logic

For an input text feature `f`, v3 computes:

1. residual feature:
   `r = normalize(f - base_feat)`
2. cosine distance to each stored prototype:
   `d_i = 1 - cosine(r, prototype_i)`

Selection rule:

- choose the task with the smallest cosine distance

OOD decision:

- if `min_i d_i > threshold`, return `"fallback"`
- else return the nearest task

The router logs the full distance dictionary during inference for debugging and
threshold analysis.

## Inference Flow

`infer_dmole_v3.py`:

1. loads all saved adapters,
2. restores the residual prototype router,
3. ensures the base feature is available,
4. encodes each prompt into a projected text feature,
5. lets the router perform OOD detection and top-1 expert selection,
6. either activates the selected adapter or disables all adapters and uses the
   base model.

Current fallback behavior in v3:

- `"fallback"` triggers generation with the adapter-free base model.

## Main Strengths and Limitations

Strengths:

- no router training loop,
- low memory overhead,
- stable and interpretable cosine-distance routing.

Limitations:

- one prototype per task may be too coarse for diverse prompt variants,
- routing performance can be sensitive to the threshold value,
- the method depends heavily on the geometry of the projected pretrained text
  space.
