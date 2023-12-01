#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20
#SBATCH --time=1-0:0:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --output=cr_%a_Credit_svm.out
#SBATCH --error=cr_%a_Credit_svm.err
#SBATCH --array=1

dataset="Credit"
ml_algorithm="SupportVectorMachineExperiment"

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
srun python3 completely_random_recommendations.py --dataset $dataset --error_type $error_type --ml_algorithm $ml_algorithm
