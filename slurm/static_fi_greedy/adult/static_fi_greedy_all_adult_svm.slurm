#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20
#SBATCH --time=3-0:0:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --output=sfg_%a_adult_svm.out
#SBATCH --error=sfg_%a_adult_svm.err
#SBATCH --array=1-4

dataset="adult.csv"
ml_algorithm="SupportVectorMachineExperiment"

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
srun python3 static_feature_importance_greedy.py --dataset $dataset --error_type $error_type --ml_algorithm $ml_algorithm
