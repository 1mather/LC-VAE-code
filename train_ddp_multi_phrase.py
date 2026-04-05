import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
import tqdm
from itertools import chain
import wandb
import random
import numpy as np
import math
from pathlib import Path
from einops import rearrange
from causalvideovae.model import *
from causalvideovae.model.ema_model import EMA
from causalvideovae.dataset.ddp_sampler import CustomDistributedSampler
from causalvideovae.dataset.video_dataset import TrainVideoDataset, ValidVideoDataset
from causalvideovae.model.utils.module_utils import resolve_str_to_obj
from causalvideovae.utils.video_utils import tensor_to_video
from causalvideovae.eval.cal_ssim import calculate_ssim

try:
    import lpips
except:
    raise Exception("Need lpips to valid.")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_logger(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        f"[rank{rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    return logger

def check_unused_params(model):
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params

def set_requires_grad_optimizer(optimizer, requires_grad):
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)

def get_exp_name(args,model_config):
    phase_suffix = ""
    time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    if hasattr(args, 'phase1_steps') and args.phase1_steps > 0:
        phase_suffix = f"-p1_{args.phase1_steps}-p2_{args.phase2_steps}"
    if hasattr(model_config, 'keep_ratio') and args.keep_ratio > 0:
        phase_suffix = f"-topk{args.keep_ratio}"
    teacher_suffix = "-tchr" if getattr(args, 'teacher_loss', False) else ""
    return f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}-sr{args.sample_rate}-fr{args.num_frames}-cons{args.consistence_weight}{phase_suffix}"

def get_current_phase(current_step, phase1_steps, phase2_steps):
    """确定当前训练阶段"""
    if current_step < phase1_steps:
        return "phase1"
    elif current_step < phase1_steps + phase2_steps:
        return "phase2"
    else:
        return "phase3"

def compute_alpha_schedule(current_step, phase1_steps, phase2_steps, schedule="cosine", max_alpha=0.9):
    """
    计算Phase 2的alpha值
    
    Args:
        current_step: 当前步数
        phase1_steps: Phase 1总步数
        phase2_steps: Phase 2总步数
        schedule: "linear" 或 "cosine"
        max_alpha: alpha的最大值
    
    Returns:
        alpha值 (0 到 max_alpha)
    """
    if current_step < phase1_steps:
        return 0.0
    elif current_step >= phase1_steps + phase2_steps:
        return max_alpha
    
    # Phase 2内的进度 (0到1)
    progress = (current_step - phase1_steps) / phase2_steps
    
    if schedule == "cosine":
        # Cosine schedule: 更平滑的过渡
        alpha = max_alpha * (1 - math.cos(progress * math.pi)) / 2
    elif schedule == "linear":
        alpha = max_alpha * progress
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return alpha

def get_phase_consistency_weight(phase, base_weight, phase1_weight_multiplier=1.0, 
                                  phase2_weight_multiplier=0.5, phase3_weight_multiplier=0.3):
    """根据阶段调整consistency weight"""
    if phase == "phase1":
        return base_weight * phase1_weight_multiplier
    elif phase == "phase2":
        return base_weight * phase2_weight_multiplier
    elif phase == "phase3":
        return base_weight * phase3_weight_multiplier
    else:
        return base_weight

def set_train(modules):
    for module in modules:
        module.train()

def set_eval(modules):
    for module in modules:
        module.eval()

def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)

def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
):
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    return filepath

def valid(global_rank, rank, model, val_dataloader, precision, args, current_step=None, save_masks=False):
    if args.eval_lpips:
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()

    
    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    flickering_list = []
    video_log = []
    num_video_log = args.eval_num_video_log
    
    # 启用通道记录（仅在需要保存mask时）
    m = model.module if isinstance(model, DDP) else model
    if save_masks and hasattr(m, 'enable_channel_recording'):
        m.enable_channel_recording(True)
        m.reset_channel_stats()
        if global_rank == 0:
            print(f"[Validation] Channel recording enabled for step {current_step}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["video"].to(rank)# vdieo max=1, min=0
            with torch.amp.autocast("cuda", dtype=precision):
                output = model(inputs)
                video_recon = output.sample #max=1.3457, min=-0.2472

            # Upload videos
            if global_rank == 0:
                for i in range(len(video_recon)):
                    if num_video_log <= 0:
                        break
                    video = tensor_to_video(video_recon[i])
                    video_log.append(video)
                    num_video_log -= 1
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            video_recon = rearrange(
                video_recon, "b c t h w -> (b t) c h w"
            ).contiguous()

            # Calculate PSNR
            mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))
            psnr = 20 * torch.log10(1 / torch.sqrt(mse))
            psnr = psnr.mean().detach().cpu().item()

            # Calculate LPIPS
            if args.eval_lpips:
                lpips_score = (
                    lpips_model.forward(inputs, video_recon)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                lpips_list.append(lpips_score)
            if False:
                ssim_score = calculate_ssim(inputs, video_recon)
                ssim_list.append(ssim_score)

            # Calculate Flickering
            gvideo_dif = video_recon[:, 1:] - inputs[:, :-1]
            rvideo_dif = inputs[:, 1:] - inputs[:, :-1]
            flickering = torch.abs(gvideo_dif - rvideo_dif).mean().detach().cpu().item()
            
            psnr_list.append(psnr)
            flickering_list.append(flickering)
            
            if global_rank == 0:
                bar.update()
            # Release gpus memory
            torch.cuda.empty_cache()
    
    # # 所有batch处理完后，保存通道统计信息
    # if save_masks and hasattr(m, 'save_channel_stats'):
    #     if global_rank == 0:
    #         # 创建保存目录
    #         mask_save_dir = Path("/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/causalvideovae/model/channel_name") / Path(get_exp_name(args, dict(**m.config))) / "channel_masks"
    #         mask_save_dir.mkdir(exist_ok=True, parents=True)
            
    #         # 保存统计信息（标记为hard-mask）
    #         mask_file = mask_save_dir / f"step_{current_step}_hard_masks.json"
    #         m.save_channel_stats(str(mask_file))
            
    #         # 打印统计信息
    #         m.print_channel_stats(top_k=30)
    #         total_calls = m.temporal_processor.channel_selection_stats.get('total_calls', 0)
    #         print(f"[Validation] Hard-mask channel statistics saved to {mask_file}")
    #         print(f"[Validation] Total samples/batches processed: {total_calls}")
        
        # 禁用通道记录
    # m.enable_channel_recording(False)
    psnr=torch.tensor(psnr_list).mean().detach().cpu().item()
    #ssim=torch.tensor(ssim_list).mean().detach().cpu().item()
    print(f"[Validation] PSNR: {psnr}")
    #print(f"[Validation] SSIM: {ssim}")
    return psnr_list, lpips_list, flickering_list, video_log

def gather_valid_result(psnr_list, lpips_list, flickering_list, video_log_list, rank, world_size):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_flickering_list = [None for _ in range(world_size)]
    gathered_video_logs = [None for _ in range(world_size)]
    
    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_flickering_list, flickering_list)
    dist.all_gather_object(gathered_video_logs, video_log_list)
    return (
        np.array(gathered_psnr_list).mean(),
        np.array(gathered_lpips_list).mean(),
        np.array(gathered_flickering_list).mean(),
        list(chain(*gathered_video_logs)),
    )

def train(args):
    # setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    logger = setup_logger(rank)
    # load generator model
    model_cls = ModelRegistry.get_model(args.model_name)

    if not model_cls:
        raise ModuleNotFoundError(
            f"`{args.model_name}` not in {str(ModelRegistry._models.keys())}."
        )

    if args.pretrained_model_name_or_path is not None:
        if global_rank == 0:
            logger.warning(
                f"You are loading a checkpoint from `{args.pretrained_model_name_or_path}`."
            )
        model = model_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    else:
        if global_rank == 0:
            logger.warning(f"Model will be inited randomly.")
        model = model_cls.from_config(args.model_config)
    
    # Prepare model config for all ranks (needed for checkpoint directory naming)
    model_config = dict(**model.config)
    args_config = dict(**vars(args))
    if 'resolution' in model_config:
        del model_config['resolution']
    
    if global_rank == 0:
        logger.warning("Connecting to WANDB...")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "causalvideovae"),
            config={**model_config, **args_config},
            name=get_exp_name(args,model_config),
        )
    
    # Create checkpoint directory on rank 0
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args,model_config))
    if global_rank == 0:
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        assert ckpt_dir.exists(), f"Checkpoint directory created: {ckpt_dir}"
        assert ckpt_dir.is_dir(), f"Checkpoint directory is directory: {ckpt_dir}"
        print(f"Checkpoint directory created: {ckpt_dir}")
        
    dist.barrier()
    
    # load discriminator model
    disc_cls = resolve_str_to_obj(args.disc_cls, append=False)
    logger.warning(
        f"disc_class: {args.disc_cls} perceptual_weight: {args.perceptual_weight}  loss_type: {args.loss_type}"
    )
    disc = disc_cls(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
        wavelet_weight=args.wavelet_weight,
        disc_factor=args.disc_factor
    )

    # DDP
    model = model.to(rank)
    
    if args.enable_tiling:
        model.enable_tiling()
    
    model = DDP(
        model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )
    disc = disc.to(rank)
    disc = DDP(
        disc, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )

    # load dataset
    dataset = TrainVideoDataset(
        args.video_path,
        sequence_length=args.num_frames,
        resolution=args.resolution,
        sample_rate=args.sample_rate,
        dynamic_sample=args.dynamic_sample,
        cache_file="idx.pkl", # Cache idx
        is_main_process=global_rank == 0,
    )
    
    ddp_sampler = CustomDistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True,
        num_workers=args.dataset_num_worker,
    )
    val_dataset = ValidVideoDataset(
        real_video_dir=args.eval_video_path,
        num_frames=args.eval_num_frames,
        sample_rate=args.eval_sample_rate,
        crop_size=args.eval_resolution,
        resolution=args.eval_resolution,
    )
    indices = range(args.eval_subset_size)
    val_dataset = Subset(val_dataset, indices=indices)
    val_sampler = CustomDistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )

    # optimizer
    modules_to_train = [module for module in model.module.get_decoder()]
    if args.freeze_encoder:
        for module in model.module.get_encoder():
            module.eval()
            module.requires_grad_(False)
        logger.info("Encoder is freezed!")
    else:
        modules_to_train += [module for module in model.module.get_encoder()]

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train += list(filter(lambda p: p.requires_grad, module.parameters()))

    gen_optimizer = torch.optim.AdamW(parameters_to_train, lr=args.lr, weight_decay=args.weight_decay)
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda')
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32
    
    # load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False)
        
        # resume optimizer
        if not args.not_resume_optimizer:
            gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])
        
        # resume discriminator
        if not args.not_resume_discriminator:
            disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
            disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
            scaler.load_state_dict(checkpoint["scaler_state"])
        
        # resume data sampler
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])
        
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()

    logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0:
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Encoder:\t{total_params(model.module.encoder):d}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.decoder):d}M")
        logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        
        # 输出三阶段训练配置
        if args.enable_three_phase:
            logger.info("=" * 60)
            logger.info("Three-Phase Training Enabled:")
            logger.info(f"  Phase 1 (Soft Consistency): 0 -> {args.phase1_steps} steps")
            logger.info(f"  Phase 2 (Soft Compression):  {args.phase1_steps} -> {args.phase1_steps + args.phase2_steps} steps")
            logger.info(f"  Phase 3 (Hard Compression):  {args.phase1_steps + args.phase2_steps} -> end")
            logger.info(f"  Phase 2 mode: {args.phase2_mode}")
            logger.info(f"  Alpha schedule: {args.alpha_schedule} (max={args.max_alpha})")
            logger.info(f"  Variance floor: {args.variance_floor}")
            logger.info(f"  Consistency weight multipliers: P1={args.phase1_weight_mult}, P2={args.phase2_weight_mult}, P3={args.phase3_weight_mult}")
            logger.info("=" * 60)
        
        logger.info("Start training!")
    
    # 初始化当前阶段
    current_phase = "phase1"
    if args.enable_three_phase:
        current_phase = get_current_phase(current_step, args.phase1_steps, args.phase2_steps)
        if hasattr(model.module, 'set_training_phase'):
            model.module.set_training_phase(current_phase)
            if global_rank == 0:
                logger.info(f"Initialized training phase: {current_phase}")

    # training bar
    bar_desc = "Epoch: {current_epoch}, Loss: {loss}"
    bar = None
    if global_rank == 0:
        max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataloader)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    num_epochs = args.epochs

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            bar.update()
    
    # training Loop
    for epoch in range(num_epochs):
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["video"].to(rank)
            
            # ========= 三阶段训练：阶段切换和参数调度 =========
            if args.enable_three_phase:
                new_phase = get_current_phase(current_step, args.phase1_steps, args.phase2_steps)
                
                # 检测阶段切换
                if new_phase != current_phase:
                    current_phase = new_phase
                    if hasattr(model.module, 'set_training_phase'):
                        model.module.set_training_phase(current_phase)
                    
                    if global_rank == 0:
                        logger.info("=" * 60)
                        logger.info(f"[Step {current_step}] Switching to {current_phase}")
                        logger.info("=" * 60)
                
                # Phase 2: 调度 alpha
                if current_phase == "phase2":
                    alpha = compute_alpha_schedule(
                        current_step, 
                        args.phase1_steps, 
                        args.phase2_steps,
                        schedule=args.alpha_schedule,
                        max_alpha=args.max_alpha
                    )
                    if hasattr(model.module, 'set_alpha'):
                        model.module.set_alpha(alpha)
                    
                    # Phase 2 progressive_frames 模式：逐步减少目标帧数
                    if args.phase2_mode == "progressive_frames" and args.phase2_target_frames is not None:
                        # 可以定义多个milestone，逐步减少帧数
                        # 这里简化为线性减少
                        phase2_progress = (current_step - args.phase1_steps) / args.phase2_steps
                        # 例如从T到target_frames线性减少
                        # 你可以根据需要调整这个策略
                        if hasattr(model.module, 'set_target_frames'):
                            model.module.set_target_frames(args.phase2_target_frames)
                
                # 动态调整consistency weight
                if args.enable_dynamic_consistency_weight:
                    dynamic_weight = get_phase_consistency_weight(
                        current_phase,
                        args.consistence_weight,
                        args.phase1_weight_mult,
                        args.phase2_weight_mult,
                        args.phase3_weight_mult
                    )
                    if hasattr(model.module, 'set_consistency_weight'):
                        model.module.set_consistency_weight(dynamic_weight)
                    # 用于loss计算
                    effective_consistency_weight = dynamic_weight
                else:
                    effective_consistency_weight = args.consistence_weight
            else:
                effective_consistency_weight = args.consistence_weight
            
            # select generator or discriminator
            if (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                step_gen = True
                step_dis = False
                
            assert (
                step_gen or step_dis
            ), "You should backward either Gen. or Dis. in a step."

            # forward
            with torch.amp.autocast('cuda', dtype=precision):
                # 在discriminator步骤时禁用generator的梯度
                if step_dis:
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    
                recon = outputs.sample
                posterior = outputs.latent_dist
                wavelet_coeffs = None
                # extra_output现用于topk辅助重建；保留wavelet_loss逻辑，避免与tensor混淆
                if args.wavelet_loss and isinstance(outputs.extra_output, (list, tuple)):
                    wavelet_coeffs = outputs.extra_output
                # 训练阶段的Top-K辅助重建（验证时为None）
                aux_recon = outputs.extra_output if (model.training and not isinstance(outputs.extra_output, (list, tuple))) else None
                
                # 获取consistency loss
                consistence = getattr(outputs, 'lowfreq_consistency_loss', None)
                if consistence is None:
                    consistence = torch.tensor(0.0, device=rank, dtype=precision, requires_grad=True)
                consistence_variance = getattr(outputs, 'lowfreq_variance', None)
                if consistence_variance is None:
                    consistence_variance = torch.tensor(0.0, device=rank, dtype=precision)
                if args.teacher_loss:
                    z_student = getattr(outputs, 'student_latents', None)
                    z_teacher = getattr(outputs, 'teacher_latents', None)
                else:
                    z_student = None
                    z_teacher = None
            # generator loss
            if step_gen:
                with torch.amp.autocast('cuda', dtype=precision):
                    g_loss, g_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=0, # 0 - generator
                        global_step=current_step,
                        consistence=consistence,
                        consistence_weight=effective_consistency_weight,  # 使用动态权重
                        last_layer=model.module.get_last_layer(),
                        wavelet_coeffs=wavelet_coeffs,
                        aux_reconstructions=aux_recon,
                        aux_weight=getattr(args, 'topk_aux_weight', 0.0),
                        split="train",
                        student_latents=z_student,
                        teacher_latents=z_teacher,
                    )
                gen_optimizer.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.step(gen_optimizer)
                scaler.update()
                
                # update ema
                if args.ema:
                    ema.update()
                    
                # log to wandb
                if global_rank == 0 and current_step % args.log_steps == 0:
                    # 基础损失指标
                    wandb.log(
                        {"train/generator_loss": g_loss.item()}, step=current_step
                    )
                    wandb.log(
                        {"train/rec_loss": g_log['train/rec_loss']}, step=current_step
                    )
                    if posterior is not None and hasattr(posterior, 'sample'):
                        wandb.log(
                            {"train/latents_std": posterior.sample().std().item()}, step=current_step
                        )
                    wandb.log(
                        {"train/consistence_loss": consistence.item()}, step=current_step
                    )
                    wandb.log(
                        {"train/consistence_loss_weighted": g_log['train/consistence_loss_weighted']}, step=current_step
                    )
                    wandb.log(
                        {"train/LLL_variance": consistence_variance.item()}, step=current_step
                    )
                    wandb.log(
                        {"train/distill_loss": g_log['train/distill_loss']}, step=current_step
                    )
                    wandb.log(
                        {"train/aux_nll_loss": g_log['train/aux_nll_loss']}, step=current_step
                    )
                    
                    # 三阶段训练相关指标
                    if args.enable_three_phase:
                        wandb.log({"training/phase": current_phase}, step=current_step)
                        wandb.log({"training/consistency_weight": effective_consistency_weight}, step=current_step)
                        
                        if current_phase == "phase2":
                            alpha = compute_alpha_schedule(
                                current_step, args.phase1_steps, args.phase2_steps,
                                schedule=args.alpha_schedule, max_alpha=args.max_alpha
                            )
                            wandb.log({"training/alpha": alpha}, step=current_step)

            # discriminator loss
            if step_dis:
                with torch.amp.autocast('cuda', dtype=precision):
                    d_loss, d_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=1,
                        global_step=current_step,
                        consistence=consistence,
                        consistence_weight=effective_consistency_weight,  # 使用动态权重
                        last_layer=None,
                        split="train",
                    )
                disc_optimizer.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.unscale_(disc_optimizer)
                scaler.step(disc_optimizer)
                scaler.update()
                if global_rank == 0 and current_step % args.log_steps == 0:
                    wandb.log(
                        {"train/discriminator_loss": d_loss.item()}, step=current_step
                    )

            update_bar(bar)
            current_step += 1

            # valid model
            
            def _set_temporal_processor_train_state(model, train_state: bool):
                m = model.module if isinstance(model, DDP) else model
                if hasattr(m, 'temporal_processor'):
                    if train_state:
                        m.temporal_processor.train()
                    else:
                        m.temporal_processor.eval()

            def valid_model(model, name="", save_channel_masks=False):
                # Soft-mask validation (training-style soft top-k)
                # _set_temporal_processor_train_state(model, True)
                # set_eval(modules_to_train)
                # psnr_list_s, lpips_list_s, flickering_list_s, video_log_s = valid(
                #     global_rank, rank, model, val_dataloader, precision, args,
                #     current_step=current_step, save_masks=False  # soft模式不保存统计
                # )
                # v_psnr_s, v_lpips_s, v_flicker_s, v_videos_s = gather_valid_result(
                #     psnr_list_s, lpips_list_s, flickering_list_s, video_log_s, rank, dist.get_world_size()
                # )
                # if global_rank == 0:
                #     suffix = ("_" + name) if name != "" else ""
                #     wandb.log(
                #         {f"val{suffix}_soft/recon": wandb.Video(np.array(v_videos_s), fps=10)},
                #         step=current_step,
                #     )
                #     wandb.log({f"val{suffix}_soft/psnr": v_psnr_s}, step=current_step)
                #     wandb.log({f"val{suffix}_soft/lpips": v_lpips_s}, step=current_step)
                #     wandb.log({f"val{suffix}_soft/flickering": v_flicker_s}, step=current_step)

                # Hard-mask validation (inference-style hard top-k)
                # 【重要】只在hard-mask模式下记录统计，这样得到的是binary mask
                _set_temporal_processor_train_state(model, False)
                set_eval(modules_to_train)
                psnr_list_h, lpips_list_h, flickering_list_h, video_log_h = valid(
                    global_rank, rank, model, val_dataloader, precision, args,
                    current_step=current_step, save_masks=save_channel_masks  # hard模式保存统计
                )
                v_psnr_h, v_lpips_h, v_flicker_h, v_videos_h = gather_valid_result(
                    psnr_list_h, lpips_list_h, flickering_list_h, video_log_h, rank, dist.get_world_size()
                )
                if global_rank == 0:
                    suffix = ("_" + name) if name != "" else ""
                    wandb.log(
                        {f"val{suffix}_hard/recon": wandb.Video(np.array(v_videos_h), fps=10)},
                        step=current_step,
                    )
                    wandb.log({f"val{suffix}_hard/psnr": v_psnr_h}, step=current_step)
                    wandb.log({f"val{suffix}_hard/lpips": v_lpips_h}, step=current_step)
                    wandb.log({f"val{suffix}_hard/flickering": v_flicker_h}, step=current_step)

                # Reset temporal processor to train for subsequent training steps
                _set_temporal_processor_train_state(model, True)

            if current_step % args.eval_steps == 0 or current_step == 1:
                if global_rank == 0:
                    logger.info("Starting validation...")
                # 根据参数决定是否保存通道masks
                # 如果设置了--save_channel_masks，则总是保存
                # 否则，仅在Phase 3时保存
                save_masks = args.save_channel_masks 
                valid_model(model, save_channel_masks=save_masks)
                if args.ema:
                    ema.apply_shadow()
                    valid_model(model, "ema", save_channel_masks=save_masks)
                    ema.restore()

            # save checkpoint
            if current_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "dics_model": disc.module.state_dict(),
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.shadow if args.ema else {},
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")
                
    # end training
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    # Exp setting
    parser.add_argument(
        "--exp_name", type=str, default="test", help="number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="number of epochs to train"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="")
    parser.add_argument("--ckpt_dir", type=str, default="./results/", help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--log_steps", type=int, default=5, help="log steps")
    parser.add_argument("--freeze_encoder", action="store_true", help="")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5, help="")

    # Data
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--num_frames", type=int, default=17, help="")
    parser.add_argument("--resolution", type=int, default=256, help="")
    parser.add_argument("--sample_rate", type=int, default=2, help="")
    parser.add_argument("--dynamic_sample", action="store_true", help="")
    # Generator model
    parser.add_argument("--ignore_mismatched_sizes", action="store_true", help="")
    parser.add_argument("--find_unused_parameters", action="store_true", help="")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help=""
    )
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument("--not_resume_training_process", action="store_true", help="")
    parser.add_argument("--enable_tiling", action="store_true", help="")
    parser.add_argument("--model_config", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="")
    parser.add_argument("--not_resume_discriminator", action="store_true", help="")
    parser.add_argument("--not_resume_optimizer", action="store_true", help="")
    parser.add_argument("--wavelet_weight", type=float, default=0.1, help="")
    parser.add_argument("--consistence_weight", type=float, default=1.0, help="")
    
    # Three-Phase Training Parameters
    parser.add_argument("--enable_three_phase", action="store_true", 
                        help="Enable three-phase temporal compression training")
    parser.add_argument("--phase1_steps", type=int, default=50000,
                        help="Number of steps for Phase 1 (soft consistency)")
    parser.add_argument("--phase2_steps", type=int, default=50000,
                        help="Number of steps for Phase 2 (soft compression)")
    parser.add_argument("--phase2_mode", type=str, default="gate", choices=["gate", "progressive_frames"],
                        help="Phase 2 compression mode: 'gate' (门控混合) or 'progressive_frames' (逐步减帧)")
    parser.add_argument("--phase2_target_frames", type=int, default=None,
                        help="Target frames for progressive_frames mode in Phase 2")
    parser.add_argument("--alpha_schedule", type=str, default="cosine", choices=["linear", "cosine"],
                        help="Alpha scheduling strategy for Phase 2")
    parser.add_argument("--max_alpha", type=float, default=0.9,
                        help="Maximum alpha value for Phase 2 gate mode")
    parser.add_argument("--variance_floor", type=float, default=0.02,
                        help="Variance floor threshold to prevent collapse")
    parser.add_argument("--enable_dynamic_consistency_weight", action="store_true",
                        help="Enable dynamic consistency weight adjustment across phases")
    parser.add_argument("--phase1_weight_mult", type=float, default=1.0,
                        help="Consistency weight multiplier for Phase 1")
    parser.add_argument("--phase2_weight_mult", type=float, default=0.5,
                        help="Consistency weight multiplier for Phase 2")
    parser.add_argument("--phase3_weight_mult", type=float, default=0.3,
                        help="Consistency weight multiplier for Phase 3")
    
    # Discriminator Model
    parser.add_argument("--load_disc_from_checkpoint", type=str, default=None, help="")
    parser.add_argument(
        "--disc_cls",
        type=str,
        default="causalvideovae.model.losses.LPIPSWithDiscriminator3D",
        help="",
    )
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")
    parser.add_argument("--loss_type", type=str, default="l1", help="")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="")
    parser.add_argument("--disc_factor", type=float, default=1.0, help="")
    parser.add_argument("--topk_aux_weight", type=float, default=1.0, help="Weight for auxiliary Top-K reconstruction loss in training")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--eval_video_path", type=str, default=None, help="")
    parser.add_argument("--eval_num_frames", type=int, default=17, help="")
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_sample_rate", type=int, default=1, help="")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="")
    parser.add_argument("--eval_num_video_log", type=int, default=2, help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")
    parser.add_argument("--save_channel_masks", action="store_true", default=False,
                        help="Save channel selection masks during validation")

    # Teacher loss toggle
    parser.add_argument("--teacher_loss", action="store_true", help="Enable teacher-student latent distillation (Top-K teacher loss)")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=4, help="")

    # EMA
    parser.add_argument("--ema", action="store_true", help="")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results/", help="")
    parser.add_argument("--wandb", action="store_true", help="")

    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)

if __name__ == "__main__":
    main()