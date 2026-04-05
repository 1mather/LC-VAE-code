import os
import sys

# Get the project root directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_wan2_path = os.path.join(_project_root, 'Wan2.1')

# Add paths
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _wan2_path not in sys.path:
    sys.path.insert(0, _wan2_path)

from .latte import Latte_models
from .latte_img import LatteIMG_models
from .latte_t2v import LatteT2V
from wan.modules.vace_model import VaceWanModel
from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'LatteIMG' in args.model:
        return LatteIMG_models[args.model](
                input_size=args.latent_size,
                in_channels=getattr(args, 'in_channels', None) or 4,
                num_classes=args.num_classes,
                num_frames=getattr(args, 'latent_num_frames', None) or args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
    elif 'LatteT2V' in args.model:
        return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)
    elif 'Latte' in args.model:
        return Latte_models[args.model](
                input_size=args.latent_size,
                in_channels=getattr(args, 'in_channels', None) or 4,
                num_classes=args.num_classes,
                num_frames=getattr(args, 'latent_num_frames', None) or args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras
            )
    elif 'VaceWanModel' in args.model:
        # Map Latte-style args to VaceWanModel parameters
        in_dim = getattr(args, 'in_channels', None) or 4
        # Use from_pretrained if path is provided, otherwise use default initialization
        if hasattr(args, 'pretrained_model_path') and args.pretrained_model_path:
            return VaceWanModel.from_pretrained(args.pretrained_model_path)
        else:
            # Initialize with 1.3B model parameters (matching wan_t2v_1_3B config)
            return VaceWanModel(
                in_dim=in_dim,
                out_dim=in_dim * 2 if getattr(args, 'learn_sigma', False) else in_dim,
                vace_layers=getattr(args, 'vace_layers', None),
                vace_in_dim=getattr(args, 'vace_in_dim', None),
                patch_size=getattr(args, 'patch_size', (1, 2, 2)),
                dim=getattr(args, 'dim', 1536),  # 1.3B config: 1536 (not 2048)
                ffn_dim=getattr(args, 'ffn_dim', 8960),  # 1.3B config: 8960 (not 8192)
                num_layers=getattr(args, 'num_layers', 30),  # 1.3B config: 30 (not 32)
                num_heads=getattr(args, 'num_heads', 12),  # 1.3B config: 12 (not 16)
            )
    else:
        raise '{} Model Not Supported!'.format(args.model)
    