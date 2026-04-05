python cal_fvd.py \
  --real_dir /scratch/cs/vidgen/guanjr/Webvid_2000 \
  --gen_dir /scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/Latte/results/latte_400k_4chan_Webvid/validation_samples/step_0380001/generated \
  --device cuda \
  --metrics is \
  --fvd_method styleganv \
  --num_frames 32 \
  --resolution 256