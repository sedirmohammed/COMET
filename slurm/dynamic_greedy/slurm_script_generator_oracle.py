from config.definitions import ROOT_DIR

def generate_slurm_script(dataset_name, ml_algorithm, abbreviation):
    slurm_template = """#!/bin/bash
#SBATCH -A naumann
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20
#SBATCH --time=5-0:0:0
#SBATCH --partition=magic
#SBATCH --constraint=ARCH:X86
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sedir.mohammed@hpi.de
#SBATCH --output=dgo_%a_{dataset_name}_{abbreviation}.out
#SBATCH --error=dgo_%a_{dataset_name}_{abbreviation}.err
#SBATCH --array=1-4

dataset="{dataset_name}"
ml_algorithm="{ml_algorithm}"

case $SLURM_ARRAY_TASK_ID in
    1)
    error_type="CategoricalShiftModifier"
    ;;
    2)
    error_type="MissingValuesModifier"
    ;;
    3)
    error_type="ScalingModifier"
    ;;
    4)
    error_type="GaussianNoiseModifier"
    ;;
esac

source /hpi/fs00/home/sedir.mohammed/miniconda3/bin/activate cleaning_recommendations
conda activate cleaning_recommendations
export PYTHONPATH="${{PYTHONPATH}}:/hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance"
cd /hpi/fs00/home/sedir.mohammed/dataquality_4ai_feature_importance/classification/greedy
srun python3 dynamic_greedy_optimal.py --dataset $dataset --error_type $error_type --ml_algorithm $ml_algorithm
"""

    formatted_script = slurm_template.format(dataset_name=dataset_name, ml_algorithm=ml_algorithm,
                                             abbreviation=abbreviation)

    with open(f"{ROOT_DIR}/slurm/dynamic_greedy/dynamic_greedy_oracle_{dataset_name}_{abbreviation}.slurm", "w") as out_file:
        out_file.write(formatted_script)


if __name__ == "__main__":
    dataset_name = "EEG.arff"
    algorithms = {
        "DecisionTreeExperiment": "decis_tree",
        "SupportVectorMachineExperiment": "svm",
        "MultilayerPerceptronExperiment": "mlp",
        "KNeighborsExperiment": "knn",
        "GradientBoostingExperiment": "gradient"
    }

    for algo, abbreviation in algorithms.items():
        generate_slurm_script(dataset_name, algo, abbreviation)
