#!/bin/bash

scripts_dir="/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/slurm/pre_pollution/south"
cd $scripts_dir
# Loop through the files in the scripts directory
for script in "$scripts_dir"/*.slurm; do
    sbatch "$script"
done

sleep 30

scripts_dir="/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/slurm/pre_pollution/adult"
cd $scripts_dir
# Loop through the files in the scripts directory
for script in "$scripts_dir"/*.slurm; do
    sbatch "$script"
done

sleep 30

scripts_dir="/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/slurm/pre_pollution/breast_cancer"
cd $scripts_dir
# Loop through the files in the scripts directory
for script in "$scripts_dir"/*.slurm; do
    sbatch "$script"
done

sleep 30

scripts_dir="/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/slurm/pre_pollution/cmc"
cd $scripts_dir
# Loop through the files in the scripts directory
for script in "$scripts_dir"/*.slurm; do
    sbatch "$script"
done

sleep 30

scripts_dir="/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/slurm/pre_pollution/letter"
cd $scripts_dir
# Loop through the files in the scripts directory
for script in "$scripts_dir"/*.slurm; do
    sbatch "$script"
done

sleep 30

scripts_dir="/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/slurm/pre_pollution/telco"
cd $scripts_dir
# Loop through the files in the scripts directory
for script in "$scripts_dir"/*.slurm; do
    sbatch "$script"
done

sleep 30

scripts_dir="/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/slurm/pre_pollution/imdb"
cd $scripts_dir
# Loop through the files in the scripts directory
for script in "$scripts_dir"/*.slurm; do
    sbatch "$script"
done