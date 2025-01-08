from json import load as load_json
from classification.experiments import *
from config.definitions import ROOT_DIR
from classification.utils.artifical_pollution import *
from classification.utils.util import get_pre_pollution_settings, load_pre_pollution_df, drop_table
from classification.utils.DatasetModifier import *
import warnings


def write_results_to_db(results_df, ds_name, experiment_str, error_type_str,  database_engine, if_exists='replace'):
    table_name = f'cleaned_data_results_{ds_name}_{experiment_str}_{error_type_str}'
    print(f'Writing results to db: {table_name}')
    drop_table(table_name, database_engine)
    try:
        results_df.to_sql(
            name=table_name,
            con=database_engine, if_exists=if_exists, index=False)
    except Exception as e:
        print(f'Error while writing results to db: {e}')



metadata_path = os.path.join(ROOT_DIR, 'metadata.json')
metadata = load_json(open(metadata_path, 'r'))
#DS_NAMES = ['Airbnb', 'cmc.data', 'Credit', 'EEG.arff', 'SouthGermanCredit.csv', 'TelcoCustomerChurn.csv', 'Titanic']
DS_NAMES = ['SouthGermanCredit.csv']
ERROR_TYPES = [CategoricalShiftModifier, MissingValuesModifier, GaussianNoiseModifier, ScalingModifier]
#ERROR_TYPES = [MissingValuesModifier]
#EXPERIMENTS = [SupportVectorMachineExperiment, KNeighborsExperiment, MultilayerPerceptronExperiment, GradientBoostingExperiment, SGDClassifierExperimentSVM, SGDClassifierExperimentLinearRegression, SGDClassifierExperimentLogisticRegression]
#EXPERIMENTS = [SGDClassifierExperimentSVM, SGDClassifierExperimentLinearRegression, SGDClassifierExperimentLogisticRegression]
EXPERIMENTS = [SupportVectorMachineExperiment]
DATABASE_URL = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
database_engine = create_engine(DATABASE_URL, echo=True, connect_args={'timeout': 2000})

for ds_name in DS_NAMES:
    print(f'Running {ds_name}')
    print('Start preprocessing')

    for error_type in ERROR_TYPES:
        print(f'Running {error_type.__name__}')

        pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)
        if pre_pollution_df.empty:
            continue
        for experiment in EXPERIMENTS:
            measurements_df = pd.DataFrame(columns=['pre_pollution_setting_id', 'real_f1', 'dataset', 'polluter', 'experiment'])
            pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine)
            for pollution_setting in pre_pollution_settings[:3]:
                pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
                print(f'Running pollution setting id {pre_pollution_setting_id}')
                ap = ArtificialPollution(metadata, str(database_engine.url), pre_pollution_setting_id, error_type)

                filtered_history_df, is_empty = ap.get_filtered_history_df(pre_pollution_df, 277712)
                if is_empty:
                    warnings.warn(f'Empty history for {ds_name}, {error_type.__name__} and random_seed 277712', UserWarning)
                    continue

                # create pollution setting, where all columns have 0.0 pollution
                pollution_setting = {'train': {}, 'test': {}, 'pre_pollution_setting_id': pre_pollution_setting_id}
                for col in metadata[ds_name]['numerical_cols'] + metadata[ds_name]['categorical_cols'] + [metadata[ds_name]['target']]:
                    pollution_setting['train'][col] = 0.0
                    pollution_setting['test'][col] = 0.0
                train_df_polluted, test_df_polluted = ap.get_current_polluted_training_and_test_df(filtered_history_df, ds_name, pollution_setting)
                print('Finished preprocessing')

                print(f'Running {experiment.__name__}')
                exp = experiment(train_df_polluted, test_df_polluted, metadata[ds_name], error_type.get_classname(), pre_pollution_setting_id)
                results = exp.run('', 'scenario', explain=False)
                new_results_row = pd.Series({'pre_pollution_setting_id': pre_pollution_setting_id,
                                             'real_f1': results[exp.name]['scoring']['macro avg']['f1-score'],
                                             'dataset': ds_name,
                                             'polluter': error_type.__name__,
                                             'experiment': experiment.__name__
                                             })
                print(new_results_row)
                measurements_df = pd.concat([measurements_df, new_results_row.to_frame().T], ignore_index=True)
            write_results_to_db(measurements_df, ds_name, experiment.__name__, error_type.__name__, database_engine, if_exists='replace')
