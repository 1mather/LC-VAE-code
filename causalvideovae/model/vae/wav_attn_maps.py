import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

def compute_wavelet_attention(latent, dwt):
    """
    Compute wavelet-based attention map from a VAE latent.

    Args:
        latent (Tensor): Latent tensor of shape (B, C, H, W)
        dwt: Wavelet transform module

    Returns:
        attn_map (Tensor): Attention map of shape (B, H, W), values in [0, 1]
    """
    B, C, H, W = latent.shape
    
    # Store original dtype and cast to float32 for wavelets processing
    original_dtype = latent.dtype
    latent_float32 = latent.to(dtype=torch.float32)

    # Apply 2D DWT to each channel
    LL, high = dwt(latent_float32)  # LL: low-frequency, high[0]: (B, 3, C, H//2, W//2)
    LH, HL, HH = high[0][:, 0], high[0][:, 1], high[0][:, 2]  # each: (B, C, H//2, W//2)

    # Compute average energy across channels
    LH_energy = LH.pow(2).mean(dim=1)  # (B, H//2, W//2)
    HL_energy = HL.pow(2).mean(dim=1)
    HH_energy = HH.pow(2).mean(dim=1)

    # Sum high-frequency energies
    HF_energy = LH_energy + HL_energy + HH_energy  # (B, H//2, W//2)

    # Upsample to match original spatial resolution
    attn_map = F.interpolate(HF_energy.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

    # Normalize to [0, 1] per sample
    attn_map = (attn_map - attn_map.amin(dim=(1, 2), keepdim=True)) / \
               (attn_map.amax(dim=(1, 2), keepdim=True) - attn_map.amin(dim=(1, 2), keepdim=True) + 1e-8)

    # Convert back to original dtype before returning
    return attn_map.to(dtype=original_dtype)  # shape: (B, H, W)


def get_mask_batch(A, l, T, timesteps):
    """
    Vectorized version of get_mask for a batch of timesteps.
    
    Args:
        A (Tensor): Wavelet attention map, shape (B, H, W), values in [0, 1]
        l (float): Lower bound (e.g., 0.1)
        T (int): Total number of timesteps
        timesteps (Tensor): Tensor of shape (B,) with values in [0, T]
    
    Returns:
        M (Tensor): Binary mask, shape (B, 1, H, W)
    """
    B, H, W = A.shape
    device = A.device
    # Ensure consistent dtype and device for calculations
    original_dtype = A.dtype
    A_float = A.to(dtype=torch.float32, device=device)
    
    # Make sure timesteps is on the same device as A
    timesteps = timesteps.to(device=device)
    t_matrix = timesteps.view(B, 1, 1).expand(B, H, W).to(dtype=torch.float32, device=device)
    
    thresholds = T * (A_float + l)  # shape (B, H, W)
    M = (thresholds >= t_matrix).float()  # shape (B, H, W)
    
    return M.unsqueeze(1).to(dtype=original_dtype, device=device)  # shape (B, 1, H, W)

def get_mask(A, l=0.1, T=1000, t=0):
    """
    Compute the binary mask M(t) for time-dependent adaptive refinement.

    Args:
        A (Tensor): Attention map of shape (B, H, W), values in [0, 1]
        l (float): Lower bound ratio (e.g., 0.1) ensuring all regions are refined at least l*T steps
        T (int): Total number of diffusion steps
        t (int): Current timestep (0 <= t <= T)

    Returns:
        M (Tensor): Binary mask of shape (B, H, W), with values {0, 1}
    """
    # Compute threshold matrix: T * (A + l)
    threshold = T * (A + l)  # shape: (B, H, W)

    # Mask is 1 where threshold >= t
    M = (threshold >= t).float()  # shape: (B, H, W)

    return M



if __name__ == "__main__":
    # Example usage
    B, C, H, W = 2, 3, 64, 64
    num_train_timesteps = 1000
    timesteps = torch.tensor([500, 800])  # Example timesteps for a batch
    T = 1000
    l = 0.3
    
    model_input = torch.randn(B, C, H, W)  # Example latent tensor
    noise = torch.randn_like(model_input)
    # original is  noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise 
    #but for simplicity of this snippet we just use
    noisy_model_input = model_input * noise


    ##mask 
    dwt = DWTForward(J=1, wave="haar")
    A = compute_wavelet_attention(latent = noisy_model_input, 
                                  dwt = dwt)
    M = get_mask_batch(A=A,
                       l=l,
                       T=num_train_timesteps,
                       timesteps=timesteps)

    #loss
    model_pred = torch.randn_like(noisy_model_input) #example output latent from the model

    target = noise - model_input
    masked_diff = M * (model_pred - target)  # shape (B, C, H, W)
    # we removed for simplicity of this snippet this which shuold be multiplied in the loss below
    # weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

    normal_loss = ((model_pred.float() - target.float()) ** 2).mean()

    our_loss = (masked_diff.pow(2)).mean()
    print(f"normal_loss {normal_loss}")
    print(f"our_loss {our_loss}")