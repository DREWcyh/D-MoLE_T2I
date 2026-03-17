"""Zero-Cost Proxy utilities for dynamic LoRA allocation."""

import logging

import deepspeed as ds
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def compute_zcp_scores(
    model_engine,
    dataloader,
    noise_scheduler,
    vae,
    text_encoder,
    args,
    device,
    weight_dtype,
):
    """Estimate transformer block saliency with a single Zero-Cost Proxy step."""
    model_engine.eval()
    vae.eval()

    for name, param in model_engine.named_parameters():
        if any(target in name for target in ["to_q", "to_k", "to_v", "to_out.0"]):
            param.requires_grad = True

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return {}

    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
    if "prompt_embeds" in batch:
        encoder_hidden_states = batch["prompt_embeds"].to(device, dtype=weight_dtype)
    else:
        attention_mask = batch.get("attention_mask")
        encoder_hidden_states = text_encoder(
            batch["input_ids"].to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            return_dict=False,
        )[0].to(dtype=weight_dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor

    timesteps = torch.randint(0, 1000, (model_input.shape[0],), device=device).long()
    noise = torch.randn_like(model_input)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

    resolution_value = args.resolution
    resolution = (
        torch.tensor([resolution_value, resolution_value])
        .repeat(model_input.shape[0], 1)
        .to(device, dtype=weight_dtype)
    )
    aspect_ratio = torch.tensor([1.0]).repeat(model_input.shape[0], 1).to(
        device,
        dtype=weight_dtype,
    )
    added_cond_kwargs = {
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }

    with torch.cuda.amp.autocast(enabled=(args.mixed_precision in ["fp16", "bf16"])):
        model_pred = model_engine(
            noisy_model_input,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        if model_pred.shape[1] > noise.shape[1]:
            model_pred = model_pred.chunk(2, dim=1)[0]

        loss = F.mse_loss(model_pred.float(), noise.float())

    model_engine.zero_grad()
    loss.backward()

    block_scores = {}
    for name, param in model_engine.named_parameters():
        if param.grad is None or "transformer_blocks" not in name:
            continue

        parts = name.split("transformer_blocks.")
        if len(parts) <= 1:
            continue

        block_idx = parts[1].split(".")[0]
        score = param.grad.detach().norm(2).item()
        block_scores[block_idx] = block_scores.get(block_idx, 0.0) + score

    model_engine.zero_grad()
    for _, param in model_engine.named_parameters():
        param.requires_grad = False

    return block_scores


def add_dmole_lora_adapter(model, lora_adapters, args, block_scores, task_idx):
    """Add one D-MoLE adapter using cumulative saliency-based block selection."""
    adapter_name = f"stage{task_idx + 1}"

    sorted_blocks = sorted(block_scores.items(), key=lambda item: item[1], reverse=True)
    total_saliency = sum(score for _, score in sorted_blocks)
    rho = getattr(args, "zcp_rho", 0.8)
    max_budget = getattr(args, "param_budget", 12)
    min_budget = 2

    top_block_indices = []
    cumulative_saliency = 0.0

    for block_idx, score in sorted_blocks:
        top_block_indices.append(str(block_idx))
        cumulative_saliency += score

        if total_saliency > 0 and (cumulative_saliency / total_saliency) >= rho:
            break
        if len(top_block_indices) >= max_budget:
            break

    if len(top_block_indices) < min_budget:
        top_block_indices = [str(block_idx) for block_idx, _ in sorted_blocks[:min_budget]]

    if ds.comm.get_rank() == 0:
        actual_rho = cumulative_saliency / total_saliency if total_saliency > 0 else 0.0
        logger.info(
            "D-MoLE %s: Selected %d blocks to reach %.2f%% saliency (target rho: %.2f%%)",
            adapter_name,
            len(top_block_indices),
            actual_rho * 100.0,
            rho * 100.0,
        )
        logger.info("Selected block indices: %s", top_block_indices)

    base_targets = ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    dynamic_target_modules = set()

    for module_name, _ in model.named_modules():
        parts = module_name.split(".")
        if "transformer_blocks" not in parts:
            continue

        block_pos = parts.index("transformer_blocks")
        if len(parts) <= block_pos + 1:
            continue

        block_idx = parts[block_pos + 1]
        if block_idx not in top_block_indices:
            continue

        for target in base_targets:
            if module_name.endswith(target):
                logical_name = ".".join(parts[block_pos:])
                dynamic_target_modules.add(logical_name)
                break

    if ds.comm.get_rank() == 0:
        logger.info(
            "D-MoLE: Successfully identified %d layers for adapter %s",
            len(dynamic_target_modules),
            adapter_name,
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        init_lora_weights="gaussian",
        target_modules=list(dynamic_target_modules),
    )

    if not lora_adapters:
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)
    else:
        model.add_adapter(adapter_name, lora_config)

    lora_adapters.append(adapter_name)
    for name, param in model.named_parameters():
        param.requires_grad = False
        if adapter_name in name:
            param.requires_grad = True

    return lora_adapters, model
