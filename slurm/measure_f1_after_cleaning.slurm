#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=25G
#SBATCH --cpus-per-task=20
#SBATCH --time=1-0:0:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --job-name=measure_f1_after_cleaning
#SBATCH --output=measure_f1_after_cleaning.out
#SBATCH --error=measure_f1_after_cleaning.err

source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${PYTHONPATH}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/classification
srun python3 measure_f1_after_cleaning.py