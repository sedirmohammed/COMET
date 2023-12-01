#!/bin/bash
#SBATCH -A naumann
#SBATCH --job-name=db_backup
#SBATCH --output=db_backup.out
#SBATCH --error=db_backup.err
#SBATCH --time=30-0:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

#cd /home/sedirmohammed/dev/dataqualiy_4ai_fi/db
source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${PYTHONPATH}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/db
while true
do
  python3 db_backup.py 
  sleep 60
done