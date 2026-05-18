#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=slow_nice
#SBATCH --exclude=bamgpu24,bamgpu26,bamgpu28
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-9%8
#SBATCH --job-name=j900_attr
#SBATCH --output=/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts/slurm_logs/%x_%A_%a.err
#
# Standardized-torch DeepLIFT/SHAP attributions for ONE cell type's 300 seqs.
# 10 array tasks x 30 seqs, max 5 concurrent (GPU bound).  Hippo_agft_venv (torch).
#
# Usage:
#   sbatch run_attr_standardtorch.sh --cell-type K562
#   sbatch run_attr_standardtorch.sh --cell-type HepG2
#   sbatch run_attr_standardtorch.sh --cell-type WTC11

SCRIPT_DIR="/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts"

module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL=300
N_TASKS=10
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))
if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi

EXTRA_ARGS="$@"

echo "=============================================="
echo "joint-900 SEAM Attr | Job $SLURM_JOB_ID | array $SLURM_ARRAY_TASK_ID/$N_TASKS | [$START:$END] | args: $EXTRA_ARGS | GPU $CUDA_VISIBLE_DEVICES | $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/SEAM_attr_standardtorch.py" --start "$START" --end "$END" $EXTRA_ARGS

echo "Done: $(date)"
