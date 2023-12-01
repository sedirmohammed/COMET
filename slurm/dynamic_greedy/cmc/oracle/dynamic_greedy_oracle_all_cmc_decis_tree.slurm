#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20
#SBATCH --time=5-0:0:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --output=dgo_%a_cmc_decis_tree.out
#SBATCH --error=dgo_%a_cmc_decis_tree.err
#SBATCH --array=1-4

dataset="cmc.data"
ml_algorithm="DecisionTreeExperiment"

case $SLURM_ARRAY_TASK_ID in
    1)
    error_type="CategoricalShiftModifier"
    job_error_type="CategoricalShiftModifier"
    ;;
    2)
    error_type="MissingValuesModifier"
    job_error_type="MissingValuesModifier"
    ;;
    3)
    error_type="ScalingModifier"
    job_error_type="ScalingModifier"
    ;;
    4)
    error_type="GaussianNoiseModifier"
    job_error_type="GaussianNoiseModifier"
    ;;
esac

source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${PYTHONPATH}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/classification/greedy
export THEANO_FLAGS="base_compiledir=theano_cache/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
srun python3 dynamic_greedy_optimal.py --dataset $dataset --error_type $error_type --ml_algorithm $ml_algorithm
rm -rf theano_cache/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}
