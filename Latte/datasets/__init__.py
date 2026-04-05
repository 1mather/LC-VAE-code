#from .sky_datasets import Sky
from .sky_datasets_video import Sky
from torchvision import transforms
from .taichi_datasets import Taichi
from . import video_transforms
#from .ucf101_datasets_wfvae import UCF101
from .ucf101_datasets import UCF101
from .ffs_datasets import FaceForensics
from .ffs_image_datasets import FaceForensicsImages
from .sky_image_datasets import SkyImages
from .ucf101_image_datasets import UCF101Images
from .taichi_image_datasets import TaichiImages
from causalvideovae.dataset.transform import ToTensorVideo, CenterCropVideo
from torchvision.transforms import Lambda, Compose, Resize

def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.data_num_frames * args.frame_interval) # 16 1

    if args.dataset == 'ffs':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensics(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ffs_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            #video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensicsImages(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        # transform_ucf101 = transforms.Compose(
        #     [
        #         ToTensorVideo(),
        #         Resize(args.image_size),
        #         CenterCropVideo(args.image_size),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        #     ]
        return UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101_img':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return UCF101Images(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return Taichi(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi_img':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return TaichiImages(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'sky':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Sky(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == 'sky_img':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return SkyImages(args, transform=transform_sky, temporal_sample=temporal_sample)
    else:
        raise NotImplementedError(args.dataset)