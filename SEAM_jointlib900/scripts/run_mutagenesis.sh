#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-17
#SBATCH --job-name=j900_mut
#SBATCH --output=/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts/slurm_logs/%x_%A_%a.err
#
# Mutagenesis libs (25K mutants @ 10% rate) for the 900 joint-lib sequences.
# 18 array tasks x 50 sequences each.  seam_venv (squid).
#
# Usage: sbatch run_mutagenesis.sh

SCRIPT_DIR="/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/SEAM_jointlib900/scripts"
source /grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/seam_venv/bin/activate
export PYTHONUNBUFFERED=1

TOTAL=900
N_TASKS=18
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))
if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi

echo "=============================================="
echo "joint-900 SEAM Mutagenesis | Job $SLURM_JOB_ID | array $SLURM_ARRAY_TASK_ID/$N_TASKS | [$START:$END] | $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/SEAM_mutagenisis.py" --start "$START" --end "$END"

echo "Done: $(date)"
