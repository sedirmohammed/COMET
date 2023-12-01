#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=10G
#SBATCH --cpus-per-task=20
#SBATCH --time=5:0:0
#SBATCH --partition magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --job-name=sfg_MissingValuesModifier_SouthGermanCredit_knn
#SBATCH --output=sfg_MissingValuesModifier_SouthGermanCredit_knn.out
#SBATCH --error=sfg_MissingValuesModifier_SouthGermanCredit_knn.err

dataset="SouthGermanCredit.csv"
error_type="MissingValuesModifier"
ml_algorithm="KNearestNeighborsExperiment"

source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${PYTHONPATH}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/classification/greedy
srun python3 static_feature_importance_greedy.py --dataset $dataset --error_type $error_type --ml_algorithm $ml_algorithm
