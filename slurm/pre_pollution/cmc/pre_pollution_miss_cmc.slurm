#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:20:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --job-name=pre_pollution_MissingValuesModifier_cmc
#SBATCH --output=pre_pollution_MissingValuesModifier_cmc.out
#SBATCH --error=pre_pollution_MissingValuesModifier_cmc.err
#SBATCH --exclude=node-21

dataset="../data/cmc.data"
error_type="MissingValuesModifier"

source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${PYTHONPATH}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/classification
srun python3 pre_pollution.py --dataset $dataset --error_type $error_type --overwrite 'y'
