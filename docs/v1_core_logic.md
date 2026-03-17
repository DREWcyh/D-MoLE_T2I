# D-MoLE v1 Core Logic

## Scope

This document summarizes the core logic implemented by the original v1 pipeline:

- `train_dmole_v1.py`
- `infer_dmole_v1.py`
- `feature_extractor.py`
- `router.py`
- `zcp_allocator.py`

## Overview

Version 1 combines three ideas:

1. Sequential LoRA expert expansion for each new task.
2. ZCP-based block selection to control where the new expert is attached.
3. An autoencoder router trained on residual text features.

The pipeline is still text-only at routing time, even though the feature extractor
keeps a legacy text-plus-vision fusion interface.

## Training Flow

### 1. Load the base PixArt components

The training script uses:

- `T5Tokenizer`
- `T5EncoderModel`
- `AutoencoderKL`
- `Transformer2DModel`

The base transformer is frozen. Only the active LoRA adapter is trainable.

### 2. Allocate a new expert

For each incoming task:

- If D-MoLE mode is enabled, `compute_zcp_scores()` estimates saliency on one
  batch.
- `add_dmole_lora_adapter()` selects the most important transformer blocks.
- A new adapter `stage{k}` is attached only to the selected modules.

If D-MoLE mode is disabled, `add_new_lora_adapter()` falls back to a standard
all-block LoRA allocation.

### 3. Build routing features

`feature_extractor.py` applies:

1. max-pooling over T5 token features,
2. a deterministic random orthogonal projection from 4096 to 512,
3. optional latent pooling for the auxiliary vision branch,
4. L2 normalization.

Important implementation note:

- The function name is `extract_and_fuse_features()`.
- In the current v1 code path it returns only the normalized text feature.

### 4. Register a base prompt feature

Before routing is used, the script encodes a generic prompt:

`A photo of a item`

Its feature is stored as `base_feat` in the router. Later routing features are
converted into residual features by subtracting this base feature.

### 5. Mentor lookup for warm-start inheritance

When task `k > 1` arrives:

- the current task feature centroid is computed,
- `router.get_top_k_experts()` compares it against all previous tasks,
- if the best match is not `fallback`, the new adapter copies the matched
  expert's LoRA weights as initialization.

This inheritance improves optimization efficiency, but it does not reduce the
nominal parameter count or the configured number of training steps by itself.

### 6. Train the task router

Each task owns one autoencoder:

- `router.add_task(task_name)`
- `router.train_ae(task_name, task_feature_matrix)`

The autoencoder is optimized with MSE reconstruction loss on residual text
features.

## Router Logic

`DMoLE_Router` stores:

- `task_aes`: one autoencoder per task,
- `base_feat`: the generic base prompt feature.

For an input feature `f`, v1 computes:

1. residual normalization:
   `r = normalize(f - base_feat) * 10`
2. reconstruction error for each task autoencoder:
   `d_i = MSE(AE_i(r), r)`
3. top-1 selection by minimum reconstruction error.

OOD decision:

- if `min_i d_i > threshold`, return `"fallback"`
- else return the task with the smallest reconstruction error

## Inference Flow

`infer_dmole_v1.py`:

1. loads all saved adapters,
2. reconstructs the router,
3. extracts a text feature for each prompt,
4. queries the autoencoder router,
5. activates the selected adapter.

Current fallback behavior in v1:

- if the router returns `"fallback"`, the script falls back to `stage1`
  rather than the adapter-free base model.

## Main Strengths and Limitations

Strengths:

- simple and intuitive routing rule,
- fully compatible with text-only inference,
- supports warm-start inheritance.

Limitations:

- autoencoders can overfit small task feature sets,
- reconstruction error is sometimes unstable as an OOD score,
- the feature extractor keeps a legacy fused interface while the effective
  routing signal is text-only.
