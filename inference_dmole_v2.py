import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import logging

from diffusers import AutoencoderKL, DDPMScheduler, PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
from peft import PeftModel

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
from router_v2 import CrossModalRouter
from feature_extractor_v2 import extract_text_features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_with_base_transformer(pipeline, prompt, **generation_kwargs):
    """
    Generate an image with all PEFT adapters disabled.

    The exact PEFT runtime API differs across versions, so this helper tries the
    common adapter-disabling paths in a safe order and restores adapter state
    afterward when needed.
    """
    transformer = pipeline.transformer

    # Newer PEFT versions commonly expose a context manager on the wrapped model.
    if hasattr(transformer, "disable_adapter") and callable(getattr(transformer, "disable_adapter")):
        with transformer.disable_adapter():
            return pipeline(prompt, **generation_kwargs).images[0]

    # Some versions expose layer toggles on the inner base model.
    inner_model = getattr(transformer, "base_model", None)
    if inner_model is not None and hasattr(inner_model, "disable_adapter_layers"):
        inner_model.disable_adapter_layers()
        try:
            return pipeline(prompt, **generation_kwargs).images[0]
        finally:
            if hasattr(inner_model, "enable_adapter_layers"):
                inner_model.enable_adapter_layers()

    # Fallback for wrappers that expose toggles directly.
    if hasattr(transformer, "disable_adapter_layers"):
        transformer.disable_adapter_layers()
        try:
            return pipeline(prompt, **generation_kwargs).images[0]
        finally:
            if hasattr(transformer, "enable_adapter_layers"):
                transformer.enable_adapter_layers()

    raise RuntimeError(
        "Fallback routing selected the base model, but this PEFT runtime does not expose "
        "a supported adapter-disable API."
    )
def parse_args():
    parser = argparse.ArgumentParser(description="Run D-MoLE PixArt-Alpha inference")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained base model"
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Directory containing trained D-MoLE tasks (task_0, task_1...) and router.bin"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to text file containing multiple prompts, one per line"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_dmole",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--validation_length",
        type=int,
        default=1e6,
        help="Number of prompts to use from file"
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda, cpu)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use half-precision inference"
    )
    parser.add_argument(
        "--router_threshold",
        type=float,
        default=0.2,
        help="Cosine-distance threshold for CrossModalRouter fallback"
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default="A photo of a item",
        help="Base prompt used to build residual text features for CrossModalRouter"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving images to directory: {args.output_dir}")
    
    log_filename = datetime.now().strftime("inference_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(args.output_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path), # 保存到文件
            logging.StreamHandler(sys.stdout) # 打印到终端
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting inference. Log saved to: {log_path}")

    logger.info(f"Saving images to directory: {args.output_dir}")
    logger.info("Loading base model components...")

    print("Loading base model components...")
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    transformer = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)

    with torch.no_grad():
        base_inputs = tokenizer(
            args.base_prompt,
            padding="max_length",
            max_length=120,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        base_t_raw = text_encoder(base_inputs)[0]
        base_t_raw = base_t_raw.to(device=device, dtype=weight_dtype)
    logger.info("Using residual text features with base prompt: %s", args.base_prompt)
    
    print(f"Detecting experts in adapter directory: {args.adapter_dir}")
    task_folders = [d for d in os.listdir(args.adapter_dir) if d.startswith("task_") and os.path.isdir(os.path.join(args.adapter_dir, d))]
    task_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    if not task_folders:
        raise ValueError(f"No task folders found in {args.adapter_dir}")

    def find_adapter_config(task_dir, adapter_name):
        """
        探测 adapter_config.json 的确切位置
        支持三种结构：
        1. task_N/transformer/stageN/ (PEFT 多专家保存的标准结构)
        2. task_N/transformer/
        3. task_N/
        """
        search_paths = [
            os.path.join(task_dir, "transformer", adapter_name),
            os.path.join(task_dir, "transformer"),
            task_dir
        ]
        for p in search_paths:
            if os.path.exists(os.path.join(p, "adapter_config.json")):
                return p
        return None

    first_task_dir = os.path.join(args.adapter_dir, task_folders[0])
    first_path = find_adapter_config(first_task_dir, "stage1")
    
    if first_path:
        print(f"Found stage1 at: {first_path}")
        transformer = PeftModel.from_pretrained(transformer, first_path, adapter_name="stage1")
    else:
        raise FileNotFoundError(f"Could not find adapter_config.json in {first_task_dir} or its subdirectories.")

    for i, task_folder in enumerate(task_folders[1:], start=2):
        adapter_name = f"stage{i}"
        task_dir = os.path.join(args.adapter_dir, task_folder)
        path = find_adapter_config(task_dir, adapter_name)
        
        if path:
            print(f"Loading {adapter_name} from: {path}")
            transformer.load_adapter(path, adapter_name=adapter_name)
        else:
            print(f"⚠️ Warning: Could not find config for {adapter_name} in {task_dir}. Skipping.")

    transformer.to(device)
    print(f"Successfully loaded {len(transformer.peft_config)} experts.")

    router = CrossModalRouter(text_dim=512, vision_dim=484).to(device)
    print("Initializing CrossModalRouter task registry...")
    for i in range(len(task_folders)):
        router.add_task(f"stage{i+1}")
    last_task_path = os.path.join(args.adapter_dir, task_folders[-1])
    router_path = os.path.join(last_task_path, "router.bin")
    
    if os.path.exists(router_path):
        router.load_state_dict(torch.load(router_path, map_location=device))
        router.eval()
        print(f"Loaded Router from {router_path}")
    else:
        raise FileNotFoundError(f"Router file not found at {router_path}")

    pipeline = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer, 
        torch_dtype=weight_dtype
    ).to(device)

    prompts = []
    if args.prompts_file is not None and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        prompts = prompts[:args.validation_length]
        print(f"Read {len(prompts)} prompts from file {args.prompts_file}")
    else:
        raise ValueError(f"File {args.prompts_file} does not exist")

    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating images"):
        with torch.no_grad():
            inputs = tokenizer(
                prompt, 
                padding="max_length", 
                max_length=120, 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.to(device)
            t_raw = text_encoder(inputs)[0]
            text_feat = extract_text_features(
                t_raw,
                base_t_input=base_t_raw,
            ).to(device=device, dtype=torch.float32)
            best_adapter = router.get_top_k_experts(
                text_feat, threshold=args.router_threshold
            )

        generation_kwargs = {
            "height": 512,              # Must match training resolution.
            "width": 512,               # Must match training resolution.
            "num_inference_steps": 30,
            "guidance_scale": 4.5,
            "generator": generator,
        }

        if best_adapter == "fallback":
            logger.warning(
                "⚠️ OOD Prompt: '%s' 未匹配到已知专家，使用不挂载 adapter 的基座模型生成",
                prompt,
            )
            route_name = "base_model"
            image = _generate_with_base_transformer(
                pipeline,
                prompt,
                **generation_kwargs,
            )
        else:
            route_name = best_adapter
            pipeline.transformer.set_adapter(best_adapter)
            image = pipeline(prompt, **generation_kwargs).images[0]

        print(f"🎯 Prompt: {prompt} -> Routing to Expert: {route_name}")
        image.save(os.path.join(args.output_dir, f"{i}_{route_name}.png"))
        
    
        
if __name__ == "__main__":
    main()
