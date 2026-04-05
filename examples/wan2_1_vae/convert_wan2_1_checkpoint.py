"""
Convert a pretrained Wan2_1 checkpoint to Wan2_1_VAE_Trainable format.

Usage:
    python convert_wan2_1_checkpoint.py \
        --input_checkpoint /path/to/vae_step_411000.pth \
        --output_checkpoint /path/to/wan2_1_trainable.ckpt \
        --config examples/wan2_1_vae/wan2_1_vae_config.json
"""

import argparse
import torch
import json
from pathlib import Path
from causalvideovae.model import ModelRegistry


def convert_checkpoint(input_path, output_path, config_path):
    """
    Convert Wan2_1 checkpoint to Wan2_1_VAE_Trainable format.
    
    The original checkpoint contains a WanVAE_ model state dict.
    We need to map it to the Wan2_1_VAE_Trainable structure.
    """
    print(f"Loading checkpoint from: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create the trainable model
    print("Creating Wan2_1_VAE_Trainable model...")
    model_cls = ModelRegistry.get_model("Wan2_1_VAE_Trainable")
    model = model_cls.from_config(config)
    
    # The state dict keys should match directly since we're using the same architecture
    # If there are any key mismatches, we'll need to handle them here
    print("Loading state dict...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    # Save the converted checkpoint
    print(f"Saving converted checkpoint to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save in the format expected by the training script
    torch.save({
        'state_dict': {
            'gen_model': model.state_dict(),
        },
        'epoch': 0,
        'current_step': 0,
    }, output_path)
    
    print("Conversion complete!")
    print(f"You can now resume training with: --resume_from_checkpoint {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert Wan2_1 checkpoint to trainable format')
    parser.add_argument('--input_checkpoint', type=str, required=True,
                        help='Path to input checkpoint (e.g., vae_step_411000.pth)')
    parser.add_argument('--output_checkpoint', type=str, required=True,
                        help='Path to output checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config JSON file')
    
    args = parser.parse_args()
    
    convert_checkpoint(args.input_checkpoint, args.output_checkpoint, args.config)


if __name__ == '__main__':
    main()

