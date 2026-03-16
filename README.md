# D-MoLE T2I Standalone Copy

This directory is a standalone copy of the D-MoLE text-to-image training and inference code migrated from:

`/home/chenyiha24/T2I-ConBench/train/train_scripts/item/dmole_t2i`

## Included

- `main.py`: continual training entrypoint
- `inference_dmole.py`: routed inference entrypoint
- `dataset.py`, `feature_extractor.py`, `router.py`, `zcp_allocator.py`: D-MoLE support modules
- `item/`: migrated item-domain training data and prior-preservation image folders
- `data/test_prompts/item/`: migrated prompt files used by the inference script
- `models/PixArt-XL-2-512x512/`: migrated pretrained PixArt model used by D-MoLE
- `ds_config/item.json`: DeepSpeed config used by the original item-domain run
- `scripts/train_dmole.sh`: adapted training launcher for this standalone directory
- `scripts/infer_dmole.sh`: adapted inference launcher for this standalone directory
- `legacy/train_pixart_dmole.py`: older monolithic D-MoLE implementation kept for reference

## Not copied

- training results and inference results

The migrated assets now live under:

- `/home/chenyiha24/dmole_t2i/item`
- `/home/chenyiha24/dmole_t2i/data/test_prompts/item`
- `/home/chenyiha24/dmole_t2i/models/PixArt-XL-2-512x512`

The shell scripts default to those new local dataset and model locations. You can still override paths with environment variables like `DATA_ROOT`, `PROMPTS_ROOT`, `MODEL_NAME`, `ADAPTER_DIR`, or `OUTPUT_DIR`.

## Quick Start

Train:

```bash
bash /home/chenyiha24/dmole_t2i/scripts/train_dmole.sh
```

Infer:

```bash
bash /home/chenyiha24/dmole_t2i/scripts/infer_dmole.sh
```

## Output Layout

- Training outputs default to:
  `/home/chenyiha24/dmole_t2i/outputs/train/dmole_without_prior_3/items_sequential`
- Inference outputs default to:
  `/home/chenyiha24/dmole_t2i/outputs/inference/dmole_without_prior_3/items_sequential`

## Notes

- The Python files were copied without changing the core D-MoLE logic.
- `main.py` and `inference_dmole.py` import sibling modules from the same directory, so they can be run directly from this standalone path.
