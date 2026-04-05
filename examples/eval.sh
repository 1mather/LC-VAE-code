EXP_NAME=WVAE_multi_wavelet_Channel_Keepratio_1.0_no_teacher_loss
SAMPLE_RATE=1
NUM_FRAMES=32
RESOLUTION=256
METRIC=psnr
SUBSET_SIZE=0
#REAL_DATASET_DIR=/scratch/shareddata/dldata/webvid-10m/dataset/val
REAL_DATASET_DIR="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/reconstruction_video_k400_val/real"
RECON_VIDEO_DIR="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/reconstruction_video_k400_val/generated"
echo $REAL_DATASET_DIR
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo $METRIC

if [[ $METRIC != "ssim" ]]; then
accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    --num_processes 1 \
    scripts/eval.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir ${RECON_VIDEO_DIR} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --crop_size ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --metric ${METRIC}
else
python scripts/eval.py \
    --mp \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir ${RECON_VIDEO_DIR}
    --device cpu \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --crop_size ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --metric ${METRIC}
fi