import argparse
import logging
import math
import os
import sys
import gc
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import deepspeed as ds
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from transformers import T5Tokenizer, T5EncoderModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, get_peft_model, PeftModel
from packaging import version

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入解耦出的模块
from dataset import DreamBoothDataset, collate_fn, tokenize_prompt, encode_prompt
from feature_extractor import extract_and_fuse_features
from router import DMoLE_Router
from zcp_allocator import compute_zcp_scores, add_dmole_lora_adapter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script for PixArt-Alpha with DreamBooth support.")

    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed configuration file path")
    parser.add_argument("--local_rank", type=int, default=-1, help="For deepspeed: local rank for distributed training on GPUs")
    parser.add_argument("--zero_stage", type=int, default=2, help="ZeRO optimization stage for DeepSpeed (0, 1, 2, 3)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model")
    parser.add_argument("--prediction_type", type=str, default=None, help="The prediction_type that shall be used for training.")
    parser.add_argument("--instance_data_dirs", type=str, default=None, required=True, help="Comma-separated list of folders")
    parser.add_argument("--instance_prompts", type=str, default=None, required=True, help="Comma-separated list of prompts")
    parser.add_argument("--class_data_dirs", type=str, default=None, required=False, help="Comma-separated list of folders")
    parser.add_argument("--class_prompts", type=str, default=None, help="Comma-separated list of prompts")
    parser.add_argument("--num_class_images", type=int, default=100, help="Minimal class images for prior preservation loss.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference.")
    parser.add_argument("--validation_images", required=False, default=None, nargs="+", help="Optional set of images to use for validation.")
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--tokenizer_max_length", type=int, default=120, required=False, help="The maximum length of the tokenizer.")
    parser.add_argument("--text_encoder_use_attention_mask", action="store_true", required=False, help="Whether to use attention mask")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images")
    parser.add_argument("--center_crop", default=False, action="store_true", help="Whether to center crop the input images")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device)")
    parser.add_argument("--max_train_steps", type=int, required=True, help="Total number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing")
    parser.add_argument("--pre_compute_text_embeddings", action="store_true", help="Whether or not to pre-compute text embeddings.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Initial learning rate")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="The scheduler type to use.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--validation_steps", type=int, default=100, help="Run validation every X steps.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images that should be generated")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="The integration to report the results")
    parser.add_argument("--output_dir", type=str, default="pixart-dreambooth-model", help="The output directory")
    parser.add_argument("--checkpointing_steps", type=int, default=200, help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Whether to use mixed precision.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs.")
    parser.add_argument("--load_transformer_path", type=str, default=None, help="Path to transformer model.")
    parser.add_argument("--lora_rank", type=int, default=32, help="The rank of the LoRA matrix.")
    parser.add_argument("--lora_target_modules", type=str, default="all", help="Comma-separated list of modules to apply LoRA to.")
    
    # D-MoLE Arguments
    parser.add_argument("--use_dmo_le", action="store_true", help="Enable Dynamic Mixture of LoRA Experts.")
    parser.add_argument("--param_budget", type=int, default=8, help="Number of Transformer blocks to allocate new LoRA experts per task.")
    parser.add_argument("--zcp_sample_ratio", type=float, default=0.01, help="Ratio of data to use for Zero-Cost Proxy evaluation.")
    parser.add_argument("--router_threshold", type=float, default=0.1, help="MSE threshold for Autoencoder Router fallback.")
    parser.add_argument("--use_inter_modal_curriculum", action="store_true", help="Enable dynamic gradient scaling between spatial and text modules.")
    parser.add_argument("--zcp_rho", type=float, default=0.8, help="Cumulative saliency ratio for dynamic layer allocation (default: 0.8)")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.instance_data_dirs = [dir.strip() for dir in args.instance_data_dirs.split(",")]
    args.instance_prompts = [prompt.strip() for prompt in args.instance_prompts.split(",")]

    if len(args.instance_data_dirs) != len(args.instance_prompts):
        raise ValueError(f"Number of instance directories ({len(args.instance_data_dirs)}) must match number of instance prompts ({len(args.instance_prompts)})")

    if args.with_prior_preservation:
        if args.class_data_dirs is None or args.class_prompts is None:
            raise ValueError("When using prior preservation in continual learning, you must specify both class_data_dirs and class_prompts")
        args.class_data_dirs = [dir.strip() for dir in args.class_data_dirs.split(",")]
        args.class_prompts = [prompt.strip() for prompt in args.class_prompts.split(",")]
        if len(args.class_data_dirs) != len(args.class_prompts):
            raise ValueError(f"Number of class directories ({len(args.class_data_dirs)}) must match number of class prompts ({len(args.class_prompts)})")
        if len(args.class_data_dirs) != len(args.instance_data_dirs):
            raise ValueError(f"Number of class directories ({len(args.class_data_dirs)}) must match number of instance directories ({len(args.instance_data_dirs)})")
    else:
        args.class_data_dirs = []
        args.class_prompts = []

    return args
    
def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

def restore_lora_to_fp32(model):
    for n, p in model.named_parameters():
        if "lora" in n:
            if p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)

def get_batch_prompt_embeds(batch, text_encoder, args, device, weight_dtype):
    if "prompt_embeds" in batch:
        return batch["prompt_embeds"].to(device=device, dtype=weight_dtype)

    attention_mask = batch.get("attention_mask")
    prompt_embeds = encode_prompt(
        text_encoder,
        batch["input_ids"].to(device),
        attention_mask.to(device) if attention_mask is not None else None,
        text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
    )
    return prompt_embeds.to(device=device, dtype=weight_dtype)

def load_base_and_all_loras(args, weight_dtype):
    model = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)

    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)

    if args.load_transformer_path is None:
        print("No LoRA adapters found. Loading base model.")
        return [], model

    lora_paths = {}
    for subdir in os.listdir(args.load_transformer_path):
        sub_path = os.path.join(args.load_transformer_path, subdir)
        if os.path.isdir(sub_path) and "adapter_config.json" in os.listdir(sub_path):
            lora_paths[subdir] = sub_path
    lora_adapters = sorted(lora_paths.keys())

    print(f"Loading {len(lora_adapters)} LoRA adapters from {args.load_transformer_path}")

    adapters = sorted(lora_paths.items())
    first_adapter, first_adapter_path = adapters[0]
    model = PeftModel.from_pretrained(model, first_adapter_path, adapter_name=first_adapter)
    for name, path in adapters[1:]:
        model.load_adapter(path, adapter_name=name)

    return lora_adapters, model

def add_new_lora_adapter(model, lora_adapters, args):
    """Fallback standard LoRA allocation (all blocks)"""
    if args.lora_target_modules == "all":
        target_modules = [
            "to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2", "linear_1", "linear_2",
        ]
    else:
        target_modules = args.lora_target_modules.split(",")

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2, init_lora_weights="gaussian",
        target_modules=target_modules, use_dora=False, use_rslora=False
    )

    adapter_name = f"stage{len(lora_adapters) + 1}"
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

def train_on_dataset(
    model_engine,
    text_encoder,
    tokenizer,
    vae,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    instance_data_dir,
    instance_prompt,
    class_data_dir,
    class_prompt,
    args,
    weight_dtype,
    device,
    timestamped_output_dir,
    start_global_step=0,
    pre_computed_encoder_hidden_states=None,
    pre_computed_class_prompt_encoder_hidden_states=None,
):
    """Train the model on a single dataset."""

    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        class_data_root=class_data_dir if args.with_prior_preservation else None,
        class_prompt=class_prompt if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=ds.comm.get_world_size(),
        rank=ds.comm.get_rank(),
        shuffle=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    max_train_steps = args.max_train_steps
    total_batch_size = args.train_batch_size * ds.comm.get_world_size() * args.gradient_accumulation_steps

    if ds.comm.get_rank() == 0:
        logger.info(f"***** Training on dataset: {instance_data_dir} with prompt: {instance_prompt} *****")
        if args.with_prior_preservation:
            logger.info(f"***** Using prior preservation with class data: {class_data_dir} and prompt: {class_prompt} *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = start_global_step
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc=f"Training on {os.path.basename(instance_data_dir)}",
        disable=not (ds.comm.get_rank() == 0),
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    model_engine.train()
    
    while global_step < start_global_step + max_train_steps:
        current_epoch = global_step // steps_per_epoch
        train_sampler.set_epoch(current_epoch)

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)

            if vae is not None:
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
            else:
                model_input = pixel_values

            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            encoder_hidden_states = get_batch_prompt_embeds(
                batch, text_encoder, args, device, weight_dtype
            )

            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if getattr(model_engine.module, 'config', model_engine.module).sample_size == 128:
                resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1).to(dtype=weight_dtype, device=model_input.device)
                aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1).to(dtype=weight_dtype, device=model_input.device)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            noisy_model_input = noisy_model_input.to(dtype=model_engine.module.dtype)
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_engine.module.dtype)
            timesteps = timesteps.to(dtype=model_engine.module.dtype)

            model_pred = model_engine(
                noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                added_cond_kwargs=added_cond_kwargs
            ).sample.chunk(2, 1)[0]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.with_prior_preservation:
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    snr = snr + 1
                mse_loss_weights = (torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            if args.with_prior_preservation:
                loss = loss + args.prior_loss_weight * prior_loss

            # 纯正的 DeepSpeed 反向传播
            model_engine.backward(loss)
            model_engine.step()

            progress_bar.update(1)
            global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logging.info(logs)
            progress_bar.set_postfix(**logs)

            restore_lora_to_fp32(model_engine)

            if global_step >= start_global_step + max_train_steps:
                break

        if global_step >= start_global_step + max_train_steps:
            break

    return global_step


def main():
    router = None
    args = parse_args()

    # Initialize DeepSpeed before anything else
    ds.init_distributed()

    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = os.path.join(args.output_dir, f"run")
    os.makedirs(timestamped_output_dir, exist_ok=True)


    # Configure logging only on the main process
    if ds.comm.get_rank() == 0:
        logging.basicConfig(
            filename=os.path.join(timestamped_output_dir, "train.log"),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        # Log all hyperparameters in a structured format
        logger.info("==========================================================")
        logger.info("                TRAINING HYPERPARAMETERS                  ")
        logger.info("==========================================================")
        hyperparams = vars(args)
        # Sort for consistent display
        for key in sorted(hyperparams.keys()):
            logger.info(f"{key}: {hyperparams[key]}")
        logger.info("==========================================================")

        # Save hyperparameters as JSON for easy reference
        with open(os.path.join(timestamped_output_dir, "hyperparameters.json"), "w") as f:
            import json
            json.dump(hyperparams, f, indent=4, sort_keys=True)

    # Set up wandb if needed
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if ds.comm.get_rank() == 0:  # Only initialize wandb on the main process
            wandb.init(project="pixart-dreambooth-continual", name=f"run_{timestamp}")
            # Log hyperparameters to wandb
            wandb.config.update(vars(args))

    # Set the training seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if ds.comm.get_rank() == 0:  # Only create directories on the main process
        if timestamped_output_dir is not None:
            os.makedirs(timestamped_output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights to half-precision
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load models and move them to the correct device based on local_rank
    device = torch.device("cuda", ds.comm.get_local_rank())

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        torch_dtype=weight_dtype
    )
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        torch_dtype=weight_dtype
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype
    )
    vae.requires_grad_(False)
    vae.to(device)

    # Pre-compute text embeddings if enabled
    if args.pre_compute_text_embeddings:
        logging.info("Pre-computing text embeddings for all datasets")

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )

            return prompt_embeds, text_inputs

        validation_prompt_negative_prompt_embeds, validation_prompt_negative_text_inputs = compute_text_embeddings("")

        # Pre-compute embeddings for all instance prompts
        pre_computed_encoder_hidden_states_list = []
        pre_computed_encoder_text_inputs_list = []

        for instance_prompt in args.instance_prompts:
            hidden_states, text_inputs = compute_text_embeddings(instance_prompt)
            pre_computed_encoder_hidden_states_list.append(hidden_states)
            pre_computed_encoder_text_inputs_list.append(text_inputs)

        # Pre-compute embeddings for all class prompts if using prior preservation
        pre_computed_class_prompt_encoder_hidden_states_list = []
        pre_computed_class_prompt_encoder_text_inputs_list = []

        if args.with_prior_preservation:
            for class_prompt in args.class_prompts:
                hidden_states, text_inputs = compute_text_embeddings(class_prompt)
                pre_computed_class_prompt_encoder_hidden_states_list.append(hidden_states)
                pre_computed_class_prompt_encoder_text_inputs_list.append(text_inputs)
        else:
            pre_computed_class_prompt_encoder_hidden_states_list = [None] * len(args.instance_prompts)
            pre_computed_class_prompt_encoder_text_inputs_list = [None] * len(args.instance_prompts)

        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states_list = [None] * len(args.instance_prompts)
        pre_computed_encoder_text_inputs_list = [None] * len(args.instance_prompts)
        pre_computed_class_prompt_encoder_hidden_states_list = [None] * len(args.instance_prompts)
        pre_computed_class_prompt_encoder_text_inputs_list = [None] * len(args.instance_prompts)
        validation_prompt_negative_prompt_embeds = None
        validation_prompt_negative_text_inputs = None

    lora_adapters, transformer = load_base_and_all_loras(args, weight_dtype)
    transformer.to(device)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * ds.comm.get_world_size()

    global_step = 0

    for dataset_idx, (instance_data_dir, instance_prompt) in enumerate(zip(args.instance_data_dirs, args.instance_prompts)):
        
        if ds.comm.get_rank() == 0:
            logger.info(f"\n>>> Task {dataset_idx+1}: {instance_prompt}")

        pre_comp_embeds = pre_computed_encoder_hidden_states_list[dataset_idx] if args.pre_compute_text_embeddings else None
        
        curr_class_embeds = None
        if args.with_prior_preservation and args.pre_compute_text_embeddings:
            curr_class_embeds = pre_computed_class_prompt_encoder_hidden_states_list[dataset_idx]
            
        class_data_dir = args.class_data_dirs[dataset_idx] if args.with_prior_preservation else None
        class_prompt = args.class_prompts[dataset_idx] if args.with_prior_preservation else None
        
        temp_dataset = DreamBoothDataset(
            instance_data_root=instance_data_dir,
            instance_prompt=instance_prompt,
            class_data_root=class_data_dir,
            class_prompt=class_prompt,
            class_num=args.num_class_images,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
            encoder_hidden_states=pre_comp_embeds,
            class_prompt_encoder_hidden_states=curr_class_embeds,
            tokenizer_max_length=args.tokenizer_max_length,
        )
        
        temp_dataloader = torch.utils.data.DataLoader(
            temp_dataset, batch_size=args.train_batch_size, shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation)
        )

        if args.use_dmo_le:
            # 使用解耦后的 dmole 模块 (难点 3)
            block_scores = compute_zcp_scores(
                transformer, temp_dataloader, noise_scheduler, vae, text_encoder, args, device, weight_dtype
            )
            lora_adapters, transformer = add_dmole_lora_adapter(transformer, lora_adapters, args, block_scores, dataset_idx)
        else:
            lora_adapters, transformer = add_new_lora_adapter(transformer, lora_adapters, args)

        current_adapter = lora_adapters[-1]
        active_adapters = [current_adapter]
        all_z_feats = []
        for batch in temp_dataloader:
            with torch.no_grad():
                prompt_embeds = get_batch_prompt_embeds(batch, text_encoder, args, device, weight_dtype)
                p_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                if args.with_prior_preservation:
                        prompt_embeds = prompt_embeds.chunk(2, dim=0)[0]
                        p_values = p_values.chunk(2, dim=0)[0]
                        
                latents = vae.encode(p_values).latent_dist.sample()

                # 使用解耦出的特征提取函数 (难点 1)
                z_feat_batch = extract_and_fuse_features(prompt_embeds, latents)

                z_feat = z_feat_batch.mean(dim=0, keepdim=True)
                all_z_feats.append(z_feat_batch.cpu())
                
        task_feature_matrix = torch.cat(all_z_feats, dim=0).to(device)
        task_centroid = task_feature_matrix.mean(dim=0, keepdim=True)
            
        if args.use_dmo_le:
            if router is None:
                actual_dim = z_feat.shape[-1]
                # 使用解耦后的 Router 模块
                router = DMoLE_Router(feature_dim=actual_dim).to(device)
                
                if ds.comm.get_rank() == 0:
                    logger.info("Initializing Router Base Feature with generic prompt...")
                base_prompt = "A photo of a item"
                with torch.no_grad():
                    text_encoder.to(device)
                    base_inputs = tokenizer(
                        base_prompt, 
                        padding="max_length", 
                        max_length=tokenizer.model_max_length, 
                        truncation=True, 
                        return_tensors="pt"
                    ).input_ids.to(device)
                    base_outputs = text_encoder(base_inputs)
                    base_t_raw = base_outputs[0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state
                    base_t_raw = base_t_raw.to(device, dtype=weight_dtype)
                    base_feat = extract_and_fuse_features(base_t_raw, latents=None)
                    router.set_base_feature(base_feat)
                    
                
            
            if dataset_idx > 0:
                # 使用 Router 进行判定决策 (难点 4)
                best_old = router.get_top_k_experts(task_centroid, threshold=args.router_threshold)
                if best_old and best_old != "fallback":
                    if ds.comm.get_rank() == 0: 
                        logger.info(f"🎯 Mentor Found: [{best_old}]. Inheriting weights for {current_adapter}...")
                    state_dict = transformer.state_dict()
                    with torch.no_grad():
                        for name, param in transformer.named_parameters():
                            if current_adapter in name:
                                mentor_param_name = name.replace(current_adapter, best_old)
                                if mentor_param_name in state_dict:
                                    param.copy_(state_dict[mentor_param_name])
                    if ds.comm.get_rank() == 0:
                        logger.info(f"Knowledge transfer complete. {current_adapter} starts from {best_old}.")
            
            # 记录当前任务并在 Proxy Training 阶段训练 AE (难点 2、5)
            router.add_task(current_adapter)
            router.train_ae(current_adapter, task_feature_matrix)

        transformer.set_adapter(current_adapter) 
        if ds.comm.get_rank() == 0:
            logger.info(f"Active Adapter set to: {current_adapter}")

        params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
        
        lr_scheduler = get_scheduler(
            args.lr_scheduler, optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * ds.comm.get_world_size(),
            num_training_steps=args.max_train_steps * ds.comm.get_world_size(),
        )

        model_engine, optimizer, _, lr_scheduler = ds.initialize(
            model=transformer, optimizer=optimizer, lr_scheduler=lr_scheduler,
            model_parameters=params_to_optimize, config=args.deepspeed
        )

        global_step = train_on_dataset(
            model_engine=model_engine,
            text_encoder=text_encoder, tokenizer=tokenizer, vae=vae,
            noise_scheduler=noise_scheduler, optimizer=optimizer, lr_scheduler=lr_scheduler,
            instance_data_dir=instance_data_dir, instance_prompt=instance_prompt,
            class_data_dir=class_data_dir,
            class_prompt=class_prompt,
            args=args, weight_dtype=weight_dtype, device=device,
            timestamped_output_dir=timestamped_output_dir,
            start_global_step=global_step,
            pre_computed_encoder_hidden_states=pre_comp_embeds,
            pre_computed_class_prompt_encoder_hidden_states=curr_class_embeds,
        )

        if ds.comm.get_rank() == 0:
            save_path = os.path.join(timestamped_output_dir, f"task_{dataset_idx}")
            os.makedirs(save_path, exist_ok=True)
            model_to_save = model_engine.module if hasattr(model_engine, "module") else model_engine
            model_to_save.save_pretrained(os.path.join(save_path, "transformer"))
            torch.save(router.state_dict(), os.path.join(save_path, "router.bin"))

        del temp_dataloader, temp_dataset, model_engine, optimizer, lr_scheduler
        torch.cuda.empty_cache()
        ds.comm.barrier()

    if ds.comm.get_rank() == 0:
        logger.info("🎉 Continual Learning Completed!")
        
if __name__ == "__main__":
    main()
