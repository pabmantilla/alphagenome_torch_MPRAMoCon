#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --job-name=j900_seam
#SBATCH --output=/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts/slurm_logs/%x_%A_%a.err
#
# SEAM clustering + MetaExplainer for ONE cell type's 300 seqs.
# 6 array tasks x 50 seqs.  seam_venv (seam).
#
# Usage:
#   sbatch run_explainer.sh --cell-type K562
#   sbatch run_explainer.sh --cell-type HepG2
#   sbatch run_explainer.sh --cell-type WTC11

SCRIPT_DIR="/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts"
source /grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/seam_venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL=300
N_TASKS=6
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))
if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi

EXTRA_ARGS="$@"

echo "=============================================="
echo "joint-900 SEAM Explainer | Job $SLURM_JOB_ID | array $SLURM_ARRAY_TASK_ID/$N_TASKS | [$START:$END] | args: $EXTRA_ARGS | $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/SEAM_explainer.py" --start "$START" --end "$END" $EXTRA_ARGS

echo "Done: $(date)"
