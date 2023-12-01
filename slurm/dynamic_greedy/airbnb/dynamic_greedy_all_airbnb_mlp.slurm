#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=50G
#SBATCH --cpus-per-task=25
#SBATCH --time=5-0:0:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --output=dg_%a_airbnb_mlp.out
#SBATCH --error=dg_%a_airbnb_mlp.err
#SBATCH --array=1

dataset="Airbnb"
ml_algorithm="MultilayerPerceptronExperiment"

case $SLURM_ARRAY_TASK_ID in
    1)
    error_type="ScalingModifier"
    job_error_type="ScalingModifier"
    ;;
esac

source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${PYTHONPATH}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/classification/greedy
export THEANO_FLAGS="base_compiledir=theano_cache/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
srun python3 dynamic_greedy.py --dataset $dataset --error_type $error_type --ml_algorithm $ml_algorithm
rm -rf theano_cache/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}
