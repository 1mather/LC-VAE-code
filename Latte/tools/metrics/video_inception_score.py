"""Inception Score (IS) from the paper "Improved techniques for training
GANs". Matches the original implementation by Salimans et al. at
https://github.com/openai/improved-gan/blob/master/inception_score/model.py"""

import numpy as np
from . import metric_utils

#----------------------------------------------------------------------------

NUM_FRAMES_IN_BATCH = {128: 128, 256: 128, 512: 64, 1024: 32}

#----------------------------------------------------------------------------

def compute_isv(opts, num_gen: int, num_splits: int, backbone: str):
    if backbone == 'c3d_ucf101':
        # Perfectly reproduced torchscript version of the original chainer checkpoint:
        # https://github.com/pfnet-research/tgan2/blob/f892bc432da315d4f6b6ae9448f69d046ef6fe01/tgan2/models/c3d/c3d_ucf101.py
        # It is a UCF-101-finetuned C3D model.
        detector_url = 'https://www.dropbox.com/s/jxpu7avzdc9n97q/c3d_ucf101.pt?dl=1'
    else:
        raise NotImplementedError(f'Backbone {backbone} is not supported.')

    num_frames = 16
    
    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        # Use the generated dataset resolution for batch size calculation
        resolution = gen_opts.dataset_kwargs.resolution
        gen_opts.dataset_kwargs.load_n_consecutive = num_frames
        gen_opts.dataset_kwargs.load_n_consecutive_random_offset = False
        gen_opts.dataset_kwargs.subsample_factor = 1
        gen_kwargs = dict()
        
        # Debug: Print dataset info
        if opts.rank == 0:
            print(f"[IS] Using generated dataset path: {gen_opts.dataset_kwargs.path}")
            print(f"[IS] Resolution: {resolution}, Batch size will be: {NUM_FRAMES_IN_BATCH[resolution] // num_frames}")
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        resolution = opts.dataset_kwargs.resolution
        gen_kwargs = dict(num_video_frames=num_frames, subsample_factor=1)
    
    batch_size = NUM_FRAMES_IN_BATCH[resolution] // num_frames

    gen_probs = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs={},
        capture_all=True, max_items=num_gen, temporal_detector=True, **gen_kwargs).get_all() # [num_gen, num_classes]

    if opts.rank != 0:
        return float('nan'), float('nan')

    # Debug: Check probability distributions
    if opts.rank == 0:
        print(f"[IS] Computed probabilities shape: {gen_probs.shape}")
        print(f"[IS] Probability stats - min: {gen_probs.min():.6f}, max: {gen_probs.max():.6f}, mean: {gen_probs.mean():.6f}")
        print(f"[IS] Probability sum per sample (should be ~1.0): min={gen_probs.sum(axis=1).min():.6f}, max={gen_probs.sum(axis=1).max():.6f}, mean={gen_probs.sum(axis=1).mean():.6f}")
        # Check entropy (low entropy = low diversity)
        entropy_per_sample = -np.sum(gen_probs * np.log(gen_probs + 1e-10), axis=1)
        print(f"[IS] Entropy per sample - min: {entropy_per_sample.min():.4f}, max: {entropy_per_sample.max():.4f}, mean: {entropy_per_sample.mean():.4f}")
        print(f"[IS] (Higher entropy = more diverse predictions, typical range: 2-4 for good videos)")

    scores = []
    np.random.RandomState(42).shuffle(gen_probs)
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        # Add small epsilon to avoid log(0)
        part = np.clip(part, 1e-10, 1.0)
        marginal = np.mean(part, axis=0, keepdims=True)
        marginal = np.clip(marginal, 1e-10, 1.0)
        kl = part * (np.log(part) - np.log(marginal))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
        
        if opts.rank == 0 and i == 0:
            print(f"[IS] First split KL divergence: {kl:.6f}, exp(KL): {np.exp(kl):.4f}")
    
    return float(np.mean(scores)), float(np.std(scores))

#----------------------------------------------------------------------------
