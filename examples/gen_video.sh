export CUDA_VISIBLE_DEVICES=0
REAL_DATASET_DIR=/scratch/cs/vidgen/data/kinetics-dataset/k400/test
EXP_NAME=WVAE_multi_wavelet_Channel_Keepratio_1.0_no_teacher_loss
SAMPLE_RATE=1
NUM_FRAMES=25
RESOLUTION=256
CKPT=/scratch/cs/vidgen/guanjr/experiment/WFVAE_Experiment/WVAE_Channel_Keepratio_1.0_no_teacher_loss-lr1.00e-05-bs2-rs256-sr1-fr25-cons0.0-p1_120000-p2_0-modetv_l1-20251023_165930/checkpoint-120000.ckpt
MODEL_CONFIG=/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/examples/WVAE_compressed_topk_multi_wavelet/wfvae-temporal-compressed-example-fix-topk-handcraft-teacher.json
SUBSET_SIZE=0

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/recon_video.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --num_workers 16 \
    --model_config ${MODEL_CONFIG} \
    --from_pretrained ${CKPT} \
    --model_name WVAE_Compressed_TopK_multi_wavelet \
    --crop_size_width 256 \
    --crop_size_height 256
    # --output_origin \
    # --crop_size ${RESOLUTION}
