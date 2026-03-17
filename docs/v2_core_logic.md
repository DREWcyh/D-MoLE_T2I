# D-MoLE v2 Core Logic

## Scope

This document summarizes the core logic implemented by the v2 pipeline:

- `train_dmole_v2.py`
- `infer_dmole_v2.py`
- `feature_extractor_v2.py`
- `router_v2.py`
- `zcp_allocator.py`

## Overview

Version 2 replaces the autoencoder router with a cross-modal projection router.
The key idea is:

- train a lightweight text-to-vision mapper for each task,
- store the real visual centroid of each task,
- route text-only prompts by comparing predicted visual features to stored
  visual centroids.

This makes routing more explicitly aligned with visual specialization.

## Training Flow

### 1. Load the base PixArt components

The backbone stays the same as v1:

- `T5Tokenizer`
- `T5EncoderModel`
- `AutoencoderKL`
- `Transformer2DModel`

Only the active LoRA adapter is trainable.

### 2. Allocate a new expert

The expert construction logic is unchanged from v1:

- ZCP estimates transformer block saliency,
- the new adapter is attached only to the selected blocks,
- old experts remain frozen.

### 3. Extract paired text and vision features

`feature_extractor_v2.py` provides two aligned representations:

- `text_features`: projected text features,
- `vision_features`: pooled latent features.

The text branch is built as:

1. max-pool T5 hidden states over tokens,
2. deterministic orthogonal projection to 512 dimensions,
3. subtract the projected base prompt feature if `base_t_input` is provided,
4. L2 normalize.

The vision branch is built as:

1. adaptive max-pool the VAE latents to `(11, 11)`,
2. flatten to 484 dimensions,
3. L2 normalize.

### 4. Mentor lookup for warm-start inheritance

For a new task, v2 computes the current text centroid and sends it to the
cross-modal router:

- if the router finds a close old task, the new LoRA adapter copies that old
  expert's weights,
- otherwise the task starts from a fresh adapter initialization.

### 5. Train one mapper per task

`CrossModalRouter` stores:

- `task_mappers`: one MLP per task,
- `task_centroids`: one frozen visual centroid per task.

For each task:

1. the visual centroid is saved as the mean of the task's visual features,
2. a small MLP is trained with MSE loss to map text features to the matching
   visual features.

Mapper architecture:

- `Linear(text_dim, hidden_dim)`
- `LayerNorm`
- `GELU`
- `Linear(hidden_dim, vision_dim)`

## Router Logic

For an inference text feature `t`, v2 computes for each task `i`:

1. predicted visual feature:
   `v_hat_i = mapper_i(t)`
2. cosine distance to the stored visual centroid:
   `d_i = 1 - cosine(v_hat_i, centroid_i)`

Selection rule:

- choose the task with the smallest cosine distance

OOD decision:

- if `min_i d_i > threshold`, return `"fallback"`
- else return the nearest task

## Inference Flow

`infer_dmole_v2.py`:

1. loads all saved adapters,
2. rebuilds the router task registry,
3. restores the router state,
4. encodes each prompt into a residual text feature using the base prompt,
5. routes the prompt through the cross-modal router,
6. either activates the selected adapter or disables all adapters and uses the
   base model.

Current fallback behavior in v2:

- `"fallback"` triggers generation with the adapter-free base model.

## Main Strengths and Limitations

Strengths:

- routing is explicitly tied to visual specialization,
- text-only inference is still supported,
- fallback can directly use the base model.

Limitations:

- the per-task mapper introduces extra training instability,
- mapper overfitting can hurt both OOD detection and expert selection,
- routing quality depends on the alignment quality between projected text space
  and latent vision space.
