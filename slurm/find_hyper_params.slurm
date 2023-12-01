#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=100G
#SBATCH --cpus-per-task=60
#SBATCH --time=1-0:0:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --job-name=find_hyper_params
#SBATCH --output=find_hyper_params.out
#SBATCH --error=find_hyper_params.err

source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${PYTHONPATH}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/classification
srun python3 find_hyper_params.py