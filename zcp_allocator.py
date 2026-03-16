import torch
import torch.nn.functional as F
import deepspeed as ds
import logging
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

def compute_zcp_scores(model_engine, dataloader, noise_scheduler, vae, text_encoder, args, device, weight_dtype):
    """[难点3: ZCP 分数计算]"""
    model_engine.eval()
    vae.eval()
    for name, param in model_engine.named_parameters():
        if any(x in name for x in ["to_q", "to_k", "to_v", "to_out.0"]):
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
    
    res_val = args.resolution
    resolution = torch.tensor([res_val, res_val]).repeat(model_input.shape[0], 1).to(device, dtype=weight_dtype)
    aspect_ratio = torch.tensor([1.0]).repeat(model_input.shape[0], 1).to(device, dtype=weight_dtype)
    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
    with torch.cuda.amp.autocast(enabled=(args.mixed_precision in ["fp16", "bf16"])):
        model_pred = model_engine(
            noisy_model_input, 
            encoder_hidden_states=encoder_hidden_states, 
            timestep=timesteps, 
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        if model_pred.shape[1] > noise.shape[1]:
            model_pred = model_pred.chunk(2, dim=1)[0]

        loss = F.mse_loss(model_pred.float(), noise.float())

    model_engine.zero_grad()
    loss.backward()
    block_scores = {}
    for name, param in model_engine.named_parameters():
        if param.grad is not None and "transformer_blocks" in name:
            parts = name.split("transformer_blocks.")
            if len(parts) > 1:
                block_idx = parts[1].split(".")[0]
                score = param.grad.detach().norm(2).item()
                block_scores[block_idx] = block_scores.get(block_idx, 0.0) + score
    model_engine.zero_grad()
    for name, param in model_engine.named_parameters():
        param.requires_grad = False
            
    return block_scores

def add_dmole_lora_adapter(model, lora_adapters, args, block_scores, task_idx):
    """
    [难点3: 确定 LoRA 专家挂载的物理层的方法] 
    修改点：采用累积显著性比例 (Cumulative Saliency Ratio) 动态选择层
    """
    adapter_name = f"stage{task_idx + 1}"
    
    # 1. 降序排列所有 block
    sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 2. 计算总能量 (Total Saliency)
    total_saliency = sum(score for _, score in sorted_blocks)
    
    # 3. 动态选择满足比例 rho 的前 m 层
    # 注意：建议在 args 中加入 zcp_rho (如 0.85) 和 max_param_budget (兜底最大层数)
    rho = getattr(args, "zcp_rho", 0.8) # 如果 args 没定义，默认取 80%
    max_budget = getattr(args, "param_budget", 12) # 以原 budget 作为上限
    min_budget = 2 # 强制保底至少选 2 层，防止任务太简单导致没层被激活
    
    top_block_indices = []
    cumulative_saliency = 0.0
    
    for block_idx, score in sorted_blocks:
        top_block_indices.append(str(block_idx))
        cumulative_saliency += score
        
        # 判定条件：达到能量比例，或者达到强制上限
        if (cumulative_saliency / total_saliency) >= rho:
            break
        if len(top_block_indices) >= max_budget:
            break

    # 4. 强制保底约束
    if len(top_block_indices) < min_budget:
        top_block_indices = [str(b[0]) for b in sorted_blocks[:min_budget]]

    if ds.comm.get_rank() == 0:
        actual_rho = cumulative_saliency / total_saliency
        logger.info(f"D-MoLE {adapter_name}: Selected {len(top_block_indices)} blocks "
                    f"to reach {actual_rho:.2%} saliency (Target Rho: {rho:.2%})")
        logger.info(f"Selected block indices: {top_block_indices}")

    # --- 后续挂载逻辑 ---
    base_targets = ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    dynamic_target_modules = set()
    
    # 使用 model.named_modules() 匹配需要挂载的物理层
    for name, module in model.named_modules():
        parts = name.split(".")
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts) > idx + 1:
                block_idx = parts[idx + 1]
                if block_idx in top_block_indices:
                    for target in base_targets:
                        if name.endswith(target):
                            # 这里构建 PEFT 需要的 logical name
                            logical_name = ".".join(parts[idx:])
                            dynamic_target_modules.add(logical_name)
                            break

    if ds.comm.get_rank() == 0:
        logger.info(f"D-MoLE: Successfully identified {len(dynamic_target_modules)} layers for adapter {adapter_name}")

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
    for n, p in model.named_parameters():
        p.requires_grad = False
        if adapter_name in n:
            p.requires_grad = True

    return lora_adapters, model
