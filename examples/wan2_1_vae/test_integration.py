"""
Test script to verify Wan2_1_VAE_Trainable integration.

This script tests:
1. Model can be created from config
2. Forward pass works correctly
3. Encoder/decoder separation works
4. Output format is correct
5. Training step works
"""

import torch
import json
from pathlib import Path
from causalvideovae.model import ModelRegistry


def test_model_creation():
    """Test that model can be created from config"""
    print("=" * 60)
    print("Test 1: Model Creation from Config")
    print("=" * 60)
    
    config_path = "examples/wan2_1_vae/wan2_1_vae_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_cls = ModelRegistry.get_model("Wan2_1_VAE_Trainable")
    model = model_cls.from_config(config)
    
    print(f"✓ Model created successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Z channels: {model.z_channels}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    return model


def test_forward_pass(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Test forward pass with dummy data"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input: [B=2, C=3, T=9, H=128, W=128]
    batch_size = 2
    num_frames = 9  # Must be 1 + 4*n for temporal caching
    height, width = 128, 128
    
    x = torch.randn(batch_size, 3, num_frames, height, width, device=device)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        outputs = model(x, sample_posterior=True, return_dict=True)
    
    print(f"✓ Forward pass successful")
    print(f"  Output type: {type(outputs).__name__}")
    print(f"  Reconstruction shape: {outputs.sample.shape}")
    print(f"  Latent dist available: {outputs.latent_dist is not None}")
    
    if outputs.latent_dist is not None:
        z = outputs.latent_dist.sample()
        print(f"  Latent shape: {z.shape}")
        expected_t = (num_frames + 3) // 4  # Temporal compression
        expected_h, expected_w = height // 8, width // 8
        print(f"  Expected latent shape: [B={batch_size}, C=16, T={expected_t}, H={expected_h}, W={expected_w}]")
    
    return outputs


def test_encoder_decoder_separation(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Test that encoder and decoder can be separated"""
    print("\n" + "=" * 60)
    print("Test 3: Encoder/Decoder Separation")
    print("=" * 60)
    
    encoder_modules = model.get_encoder()
    decoder_modules = model.get_decoder()
    
    print(f"✓ Encoder modules: {len(encoder_modules)}")
    for i, module in enumerate(encoder_modules):
        params = sum(p.numel() for p in module.parameters())
        print(f"    {i+1}. {type(module).__name__}: {params / 1e6:.2f}M params")
    
    print(f"✓ Decoder modules: {len(decoder_modules)}")
    for i, module in enumerate(decoder_modules):
        params = sum(p.numel() for p in module.parameters())
        print(f"    {i+1}. {type(module).__name__}: {params / 1e6:.2f}M params")
    
    # Test last layer
    last_layer = model.get_last_layer()
    print(f"✓ Last layer shape: {last_layer.shape}")


def test_training_step(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Test a training step with gradients"""
    print("\n" + "=" * 60)
    print("Test 4: Training Step with Gradients")
    print("=" * 60)
    
    model = model.to(device)
    model.train()
    
    # Create optimizer for decoder only (common in VAE training)
    decoder_params = []
    for module in model.get_decoder():
        decoder_params.extend(module.parameters())
    
    optimizer = torch.optim.Adam(decoder_params, lr=1e-4)
    
    # Create dummy input
    x = torch.randn(1, 3, 9, 128, 128, device=device)
    
    # Forward pass
    outputs = model(x, sample_posterior=True, return_dict=True)
    reconstruction = outputs.sample
    
    # Simple L1 loss
    loss = torch.nn.functional.l1_loss(reconstruction, x)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradients computed: {any(p.grad is not None for p in decoder_params)}")


def test_encode_decode_separately(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Test encoding and decoding separately"""
    print("\n" + "=" * 60)
    print("Test 5: Separate Encode/Decode")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(1, 3, 9, 128, 128, device=device)
    
    with torch.no_grad():
        # Encode
        enc_output = model.encode(x, return_dict=True)
        z = enc_output.latent_dist.mode()  # Use mode instead of sampling
        
        # Decode
        dec_output = model.decode(z, return_dict=True)
        reconstruction = dec_output.sample
    
    print(f"✓ Separate encode/decode successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Shapes match: {x.shape == reconstruction.shape}")


def main():
    print("Testing Wan2_1_VAE_Trainable Integration")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model, device)
        
        # Test 3: Encoder/decoder separation
        test_encoder_decoder_separation(model, device)
        
        # Test 4: Training step
        test_training_step(model, device)
        
        # Test 5: Separate encode/decode
        test_encode_decode_separately(model, device)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nThe model is ready to use with train_ddp_multi_phrase.py")
        print("\nNext steps:")
        print("1. Edit examples/wan2_1_vae/train_wan2_1_vae.sh to set your data paths")
        print("2. Run: bash examples/wan2_1_vae/train_wan2_1_vae.sh")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Test failed!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

