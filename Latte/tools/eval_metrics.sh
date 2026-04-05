#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash Latte/tools/eval_metrics.sh
#
# What it does:
# - Finds all `validation_samples/step_*/generated` folders under RUN_DIR
# - Runs `Latte/tools/compute_metrics.py` for each step
# - Writes per-step logs + a CSV summary (step,is_mean,is_std,fvd)
#
# Notes:
# - `compute_metrics.py` deletes its temp extraction directory at the end.
# - Set FORCE_REEXTRACT=1 to re-extract frames each step (slow but safest).

# -----------------------
# Config (edit these)
# -----------------------
REAL_VIDEO_DIR="/scratch/cs/vidgen/guanjr/Webvid_2000"
RUN_DIR="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/Latte/evaluation_curve/web_16_curve/000-Latte-XL-2-F8S3-sky/validation_samples"

MAX_VIDEOS=2000
NUM_FRAMES=16
RESOLUTION=256
NUM_WORKERS=64

# FVD temporal subsampling (set to 1 for strict consecutive frames)
FVD_REAL_SUBSAMPLE_FACTOR=1
FVD_GEN_SUBSAMPLE_FACTOR=1

# 0/1
FORCE_REEXTRACT=1

# Output files
OUT_DIR="${RUN_DIR}/metrics"
CSV_PATH="${OUT_DIR}/metrics.csv"
LOG_DIR="${OUT_DIR}/logs"

# -----------------------
# Validate
# -----------------------
if [[ ! -d "${REAL_VIDEO_DIR}" ]]; then
  echo "ERROR: REAL_VIDEO_DIR does not exist: ${REAL_VIDEO_DIR}" >&2
  exit 1
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# CSV header
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "step,is_mean,is_std,fvd,gen_dir" > "${CSV_PATH}"
fi

# -----------------------
# Main loop
# -----------------------
shopt -s nullglob
step_dirs=( "${RUN_DIR}"/step_* )

if [[ ${#step_dirs[@]} -eq 0 ]]; then
  echo "No step_* dirs found under: ${RUN_DIR}" >&2
  exit 1
fi

for step_dir in "${step_dirs[@]}"; do
  gen_dir="${step_dir}/generated"
  if [[ ! -d "${gen_dir}" ]]; then
    echo "Skipping (no generated/): ${step_dir}"
    continue
  fi

  step_name="$(basename "${step_dir}")"        # e.g., step_0060000
  step="${step_name#step_}"                    # e.g., 0060000
  log_path="${LOG_DIR}/${step_name}.log"

  echo "================================================================================"
  echo "Step ${step}: evaluating ${gen_dir}"
  echo "Log: ${log_path}"
  echo "================================================================================"

  # Build command
  cmd=( python Latte/tools/compute_metrics.py
    --real_video_dir "${REAL_VIDEO_DIR}"
    --generated_video_dir "${gen_dir}"
    --max_videos "${MAX_VIDEOS}"
    --num_frames "${NUM_FRAMES}"
    --resolution "${RESOLUTION}"
    --num_workers "${NUM_WORKERS}"
    --fvd_real_subsample_factor "${FVD_REAL_SUBSAMPLE_FACTOR}"
    --fvd_gen_subsample_factor "${FVD_GEN_SUBSAMPLE_FACTOR}"
  )
  if [[ "${FORCE_REEXTRACT}" -eq 1 ]]; then
    cmd+=( --force_reextract )
  fi

  # Run + capture log
  ( "${cmd[@]}" 2>&1 | tee "${log_path}" )

  # Extract metrics from log (robust to formatting)
  is_line="$(grep -E 'Inception Score:' "${log_path}" | tail -n 1 || true)"
  fvd_line="$(grep -E 'FVD Score:' "${log_path}" | tail -n 1 || true)"

  # Defaults
  is_mean="nan"
  is_std="nan"
  fvd="nan"

  if [[ -n "${is_line}" ]]; then
    # Example: "Inception Score: 7.1420 ± 0.1234"
    is_mean="$(echo "${is_line}" | awk '{print $3}' | tr -d '\r')"
    is_std="$(echo "${is_line}"  | awk '{print $5}' | tr -d '\r')"
  fi
  if [[ -n "${fvd_line}" ]]; then
    # Example: "FVD Score: 466.1234"
    fvd="$(echo "${fvd_line}" | awk '{print $3}' | tr -d '\r')"
  fi

  echo "${step},${is_mean},${is_std},${fvd},${gen_dir}" >> "${CSV_PATH}"
  echo "Recorded: step=${step} is=${is_mean}±${is_std} fvd=${fvd}"
done

echo "Done."
echo "CSV: ${CSV_PATH}"