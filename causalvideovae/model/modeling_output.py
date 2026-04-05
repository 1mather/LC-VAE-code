from dataclasses import dataclass
from typing import Optional
import torch
from diffusers.utils import BaseOutput

from .utils.distrib_utils import DiagonalGaussianDistribution

@dataclass
class AutoencoderKLOutput(BaseOutput):
    latent_dist: DiagonalGaussianDistribution
    extra_output: Optional[tuple] = None

@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None
    extra_output: Optional[tuple] = None
    
@dataclass
class ForwardOutput(BaseOutput):
    sample: torch.Tensor
    latent_dist: DiagonalGaussianDistribution
    lowfreq_variance: torch.Tensor = None
    lowfreq_consistency_loss: torch.Tensor = None
    extra_output: Optional[tuple] = None
    # Optional teacher-student latents for distillation (e.g., TopK teacher loss)
    teacher_latents: Optional[torch.Tensor] = None
    student_latents: Optional[torch.Tensor] = None