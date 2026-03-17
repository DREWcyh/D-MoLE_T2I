"""Inference entry point for the v3 residual prototype D-MoLE router."""

import argparse
import logging
import os
import sys
from datetime import datetime

import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from peft import PeftModel
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_extractor_v3 import extract_text_features
from router_v3 import ResidualPrototypeRouter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _generate_with_base_transformer(pipeline, prompt, **generation_kwargs):
    """Generate an image with all PEFT adapters disabled."""
    transformer = pipeline.transformer

    if hasattr(transformer, "disable_adapter") and callable(getattr(transformer, "disable_adapter")):
        with transformer.disable_adapter():
            return pipeline(prompt, **generation_kwargs).images[0]

    inner_model = getattr(transformer, "base_model", None)
    if inner_model is not None and hasattr(inner_model, "disable_adapter_layers"):
        inner_model.disable_adapter_layers()
        try:
            return pipeline(prompt, **generation_kwargs).images[0]
        finally:
            if hasattr(inner_model, "enable_adapter_layers"):
                inner_model.enable_adapter_layers()

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
    """Parse CLI arguments for v3 inference."""
    parser = argparse.ArgumentParser(description="Run D-MoLE PixArt-Alpha inference")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained base model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Directory containing trained D-MoLE tasks and router state files",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file containing one prompt per line",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_dmole",
        help="Directory used to save generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )
    parser.add_argument(
        "--validation_length",
        type=int,
        default=10**6,
        help="Maximum number of prompts to read from file",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for inference",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable half-precision inference",
    )
    parser.add_argument(
        "--router_threshold",
        type=float,
        default=0.15,
        help="Cosine-distance threshold for ResidualPrototypeRouter fallback",
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default="A photo of a item",
        help="Base prompt used for residual stripping",
    )
    return parser.parse_args()


def _configure_logging(output_dir):
    """Configure file and stdout logging for one inference run."""
    os.makedirs(output_dir, exist_ok=True)
    log_filename = datetime.now().strftime("infer_session_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(output_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def _find_adapter_config(task_dir, adapter_name):
    """Find the directory that contains the adapter config for one expert."""
    search_paths = [
        os.path.join(task_dir, "transformer", adapter_name),
        os.path.join(task_dir, "transformer"),
        task_dir,
    ]
    for candidate in search_paths:
        if os.path.exists(os.path.join(candidate, "adapter_config.json")):
            return candidate
    return None


def _load_task_folders(adapter_dir):
    """List and sort task folders under one adapter directory."""
    task_folders = [
        folder
        for folder in os.listdir(adapter_dir)
        if folder.startswith("task_") and os.path.isdir(os.path.join(adapter_dir, folder))
    ]
    task_folders.sort(key=lambda name: int(name.split("_")[1]))
    if not task_folders:
        raise ValueError(f"No task folders found in {adapter_dir}")
    return task_folders


def _load_transformer_with_adapters(args, weight_dtype):
    """Load the base transformer and attach all saved LoRA experts."""
    transformer = Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )

    task_folders = _load_task_folders(args.adapter_dir)
    print(f"Detecting experts in adapter directory: {args.adapter_dir}")

    first_task_dir = os.path.join(args.adapter_dir, task_folders[0])
    first_adapter_path = _find_adapter_config(first_task_dir, "stage1")
    if first_adapter_path is None:
        raise FileNotFoundError(
            f"Could not find adapter_config.json in {first_task_dir} or its subdirectories."
        )

    print(f"Found stage1 at: {first_adapter_path}")
    transformer = PeftModel.from_pretrained(
        transformer,
        first_adapter_path,
        adapter_name="stage1",
    )

    for index, task_folder in enumerate(task_folders[1:], start=2):
        adapter_name = f"stage{index}"
        task_dir = os.path.join(args.adapter_dir, task_folder)
        adapter_path = _find_adapter_config(task_dir, adapter_name)

        if adapter_path is None:
            print(f"Warning: Could not find config for {adapter_name} in {task_dir}. Skipping.")
            continue

        print(f"Loading {adapter_name} from: {adapter_path}")
        transformer.load_adapter(adapter_path, adapter_name=adapter_name)

    transformer.to(args.device)
    print(f"Successfully loaded {len(transformer.peft_config)} experts.")
    return transformer, task_folders


def _read_prompts(prompts_file, limit):
    """Read prompts from file and apply the requested limit."""
    if prompts_file is None or not os.path.exists(prompts_file):
        raise ValueError(f"File {prompts_file} does not exist")

    with open(prompts_file, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle.readlines() if line.strip()]
    return prompts[:limit]


def _find_router_state_path(task_dir):
    """Resolve the router state path while remaining compatible with old runs."""
    for filename in ["router_state.bin", "router.bin"]:
        candidate = os.path.join(task_dir, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Router state file not found in {task_dir}")


def _format_image_filename(sample_index, route_name):
    """Build a clearer image filename for inference outputs."""
    return f"sample_{sample_index:03d}_route-{route_name}.png"


def main():
    """Run v3 prompt routing and image generation."""
    args = parse_args()
    device = args.device
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    print(f"Saving images to directory: {args.output_dir}")
    log_path = _configure_logging(args.output_dir)
    logger.info("Starting inference. Log saved to: %s", log_path)
    logger.info("Saving images to directory: %s", args.output_dir)
    logger.info("Loading base model components...")

    print("Loading base model components...")
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    ).to(device)
    transformer, task_folders = _load_transformer_with_adapters(args, weight_dtype)

    with torch.no_grad():
        base_inputs = tokenizer(
            args.base_prompt,
            padding="max_length",
            max_length=120,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        base_hidden_states = text_encoder(base_inputs)[0]
        base_hidden_states = base_hidden_states.to(device=device, dtype=weight_dtype)
    base_text_feature = extract_text_features(base_hidden_states).to(
        device=device,
        dtype=torch.float32,
    )
    logger.info("Using residual prototype routing with base prompt: %s", args.base_prompt)

    router = ResidualPrototypeRouter(feature_dim=512).to(device)
    print("Initializing ResidualPrototypeRouter...")
    last_task_path = os.path.join(args.adapter_dir, task_folders[-1])
    router_path = _find_router_state_path(last_task_path)
    router.load_state_dict(torch.load(router_path, map_location=device))
    if not router.has_base_feat:
        router.set_base_feature(base_text_feature)
    router.eval()
    print(f"Loaded Router from {router_path}")

    pipeline = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=weight_dtype,
    ).to(device)

    prompts = _read_prompts(args.prompts_file, args.validation_length)
    print(f"Read {len(prompts)} prompts from file {args.prompts_file}")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    for index, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating images"):
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=120,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            text_hidden_states = text_encoder(inputs)[0]
            text_feature = extract_text_features(text_hidden_states).to(
                device=device,
                dtype=torch.float32,
            )
            best_adapter = router.get_top_k_experts(
                text_feature,
                threshold=args.router_threshold,
            )

        generation_kwargs = {
            "height": 512,
            "width": 512,
            "num_inference_steps": 30,
            "guidance_scale": 4.5,
            "generator": generator,
        }

        if best_adapter == "fallback":
            logger.warning(
                "OOD prompt '%s' did not match a known expert. Using the base model.",
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

        print(f"Prompt: {prompt} -> Routing to Expert: {route_name}")
        image.save(os.path.join(args.output_dir, _format_image_filename(index, route_name)))


if __name__ == "__main__":
    main()
