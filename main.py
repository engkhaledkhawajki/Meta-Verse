from Codes.Visualiztion.Visaliztion import FLTrainingResultsVisualizer
from Codes.Federated_learning.Fed_learning import FL
from Codes.Client_Server.Server import Server
import numpy as np
import config as cfg
import pickle
import os

cfg.changable_variables['classes'] = np.arange(0, cfg.basic_variables['number_of_classes'], 1.)


def count_sum_based_dict_of_tds(tds):
    V = []
    keys = list(tds.keys())
    for key in keys:
        label_for_each_user = []
        for la in list(tds[key]):
            for label in la[1]:
                label_for_each_user.append(label.numpy())
        V.append(np.sum([label_nes for label_nes in label_for_each_user], axis=0))
    V = np.array(V)
    return V


def visualize_histograms_for_clusters(clients_data=None,
                                      visualization_title=None,
                                      groups_over_rounds=None,
                                      classes=None):
    plotting = FLTrainingResultsVisualizer(visualization_title, cfg.changable_variables['base_folder_groups'],
                                           cfg.basic_variables['number_of_users'])

    plotting.extract_groups_info(groups_over_rounds)
    plotting.plot_clusters_bars(classes, clients_data)


def visualize_other_metrics_for_models(plotting,
                                       ):
    plotting.plot_other_metrics()


def extract_metrics_for_plots(plotting,
                              fl_accuracy_loss_model_with_dp_without_icg=None,
                              fl_accuracy_loss_model_without_dp_icg=None,
                              fl_accuracy_loss_model_with_icg_dp=None,
                              fl_accuracy_loss_model_with_icg_without_dp=None,
                              cen_global_model_tuple=None,
                              ):
    plotting.extract_global_metrics(
        fl_clients_round_accuracy_with_dp_without_icg=fl_accuracy_loss_model_with_dp_without_icg,
        fl_clients_round_accuracy_without_dp_icg=fl_accuracy_loss_model_without_dp_icg,
        fl_clients_round_accuracy_with_icg_dp=fl_accuracy_loss_model_with_icg_dp,
        fl_clients_round_accuracy_with_icg_without_dp=fl_accuracy_loss_model_with_icg_without_dp,
        cen_global_model_tuple=cen_global_model_tuple)

    plotting.extract_clients_metrics(
        fl_clients_round_accuracy_with_dp_without_icg=fl_accuracy_loss_model_with_dp_without_icg,
        fl_clients_round_accuracy_without_dp_icg=fl_accuracy_loss_model_without_dp_icg,
        fl_clients_round_accuracy_with_icg_dp=fl_accuracy_loss_model_with_icg_dp,
        fl_clients_round_accuracy_with_icg_without_dp=fl_accuracy_loss_model_with_icg_without_dp)


def visualize_results_for_models(plotting=None,
                                 models=True,
                                 clients=False,
                                 ):
    if models and not clients:
        plotting.plot_loss()
        plotting.plot_accuracy()
    if clients and not models:
        plotting.plot_loss_clients()
        plotting.plot_accuracy_clients()
    if models and clients:
        plotting.plot_loss()
        plotting.plot_accuracy()
        plotting.plot_loss_clients()
        plotting.plot_accuracy_clients()


def loading_results(base_folder_results):
    fl_accuracy_loss_model_with_icg_dp = None
    fl_accuracy_loss_model_with_dp_without_icg = None
    fl_accuracy_loss_model_with_icg_without_dp = None
    fl_accuracy_loss_model_without_dp_icg = None
    if cfg.base_directories['differential_privacy']:
        if cfg.base_directories['clustering']:
            a_file = open(f'{base_folder_results}/fl_accuracy_loss_model_with_icg_dp_customized_clustering.pkl', "rb")
            fl_accuracy_loss_model_with_icg_dp = pickle.load(a_file)
        else:
            a_file = open(f'{base_folder_results}/fl_accuracy_loss_model_with_dp_without_icg_customized_clustering.pkl',
                          "rb")
            fl_accuracy_loss_model_with_dp_without_icg = pickle.load(a_file)
    else:
        if cfg.base_directories['clustering']:
            a_file = open(f'{base_folder_results}/fl_accuracy_loss_model_with_icg_without_dp_customized_clustering.pkl',
                          "rb")
            fl_accuracy_loss_model_with_icg_without_dp = pickle.load(a_file)
        else:
            a_file = open(f'{base_folder_results}/fl_accuracy_loss_model_without_dp_icg_customized_clustering.pk', "rb")
            fl_accuracy_loss_model_without_dp_icg = pickle.load(a_file)
    return fl_accuracy_loss_model_with_icg_dp, fl_accuracy_loss_model_with_dp_without_icg, fl_accuracy_loss_model_with_icg_without_dp, fl_accuracy_loss_model_without_dp_icg


def call_fl(dp=False, icg=False, experiment_type=cfg.basic_variables['gen_type']):
    """
    Call FedAvg and feed - forward the model to server. Args : client_batched : A list of clients to be federated
    """
    """
        Call FedAvg algorithm and save the model and the dictionary for results

        :param dp: a boolean variable that is used to decide whether the user want the training process with Differential Privacy or not
        :param icg: a boolean variable that is used to decide whether the user want the training process with Inter Clustering Grouping or not
        :returns:
    """

    server1 = Server(cfg.basic_variables['input_shape'],
                     cfg.basic_variables['data_name'],
                     cfg.basic_variables['number_of_classes'])

    fed_learning = FL(cfg.basic_variables['number_of_users'],
                      cfg.basic_variables['comms_round'],
                      cfg.basic_variables['metrics'],
                      cfg.basic_variables['loss'],
                      cfg.basic_variables['optimizer'],
                      server1,
                      cfg.basic_variables['clipping_norm'],
                      cfg.basic_variables['epsilon'],
                      cfg.basic_variables['USERS_PER_ROUND'],
                      cfg.basic_variables['learning_rate'],
                      cfg.basic_variables['momentum'],
                      cfg.basic_variables['non_iid_strength_factor'],
                      cfg.basic_variables['sigma'],
                      cfg.changable_variables['classes'],
                      cfg.basic_variables['magic_value']
                      )

    cfg.changable_variables['clients_data'] = fed_learning.prepare_fl(cfg.data_paths['dataset_path'],
                                                                      cfg.data_paths['test_dataset_path'],
                                                                      cfg.basic_variables['batch_size'],
                                                                      cfg.basic_variables['data_name'],
                                                                      cfg.basic_variables['input_shape'],
                                                                      cfg.basic_variables['number_of_classes'],
                                                                      cfg.basic_variables['gen_type'],
                                                                      cfg.basic_variables['the_allowed_hete'])

    if experiment_type == 'extreme':
        if not os.path.exists(cfg.base_directories['base_folder_extreme_results']):
            os.mkdir(cfg.base_directories['base_folder_extreme_results'])

        if not os.path.exists(cfg.base_directories['base_folder_extreme_models']):
            os.mkdir(cfg.base_directories['base_folder_extreme_models'])

        if not os.path.exists(cfg.base_directories['base_folder_extreme_groups']):
            os.mkdir(cfg.base_directories['base_folder_extreme_groups'])

        if not os.path.exists(cfg.base_directories['base_folder_extreme_results_global_models']):
            os.mkdir(cfg.base_directories['base_folder_extreme_results_global_models'])

        if not os.path.exists(cfg.base_directories['base_folder_extreme_results_clients_models']):
            os.mkdir(cfg.base_directories['base_folder_extreme_results_clients_models'])

        cfg.changable_variables['base_folder_results'] = cfg.base_directories['base_folder_extreme_results']
        cfg.changable_variables['base_folder_models'] = cfg.base_directories['base_folder_extreme_models']
        cfg.changable_variables['base_folder_groups'] = cfg.base_directories['base_folder_extreme_groups']
        cfg.changable_variables['base_folder_results_global_models'] = cfg.base_directories[
            'base_folder_extreme_results_global_models']
        cfg.changable_variables['base_folder_results_clients_models'] = cfg.base_directories[
            'base_folder_extreme_results_clients_models']
        base_folder_clients_data = 'empty for now'

    elif experiment_type == 'hetero':
        if not os.path.exists(cfg.base_directories['base_folder_hetero_results']):
            os.mkdir(cfg.base_directories['base_folder_hetero_results'])

        if not os.path.exists(cfg.base_directories['base_folder_hetero_models']):
            os.mkdir(cfg.base_directories['base_folder_hetero_models'])

        if not os.path.exists(cfg.base_directories['base_folder_hetero_groups']):
            os.mkdir(cfg.base_directories['base_folder_hetero_groups'])

        if not os.path.exists(cfg.base_directories['base_folder_hetero_results_global_models']):
            os.mkdir(cfg.base_directories['base_folder_hetero_results_global_models'])

        if not os.path.exists(cfg.base_directories['base_folder_hetero_results_clients_models']):
            os.mkdir(cfg.base_directories['base_folder_hetero_results_clients_models'])

        cfg.changable_variables['base_folder_results'] = cfg.base_directories['base_folder_hetero_results']
        cfg.changable_variables['base_folder_models'] = cfg.base_directories['base_folder_hetero_models']
        cfg.changable_variables['base_folder_groups'] = cfg.base_directories['base_folder_hetero_groups']
        cfg.changable_variables['base_folder_results_global_models'] = cfg.base_directories[
            'base_folder_hetero_results_global_models']
        cfg.changable_variables['base_folder_results_clients_models'] = cfg.base_directories[
            'base_folder_hetero_results_clients_models']
        base_folder_clients_data = 'empty for now'

    elif experiment_type == 'homo':
        if not os.path.exists(cfg.base_directories['base_folder_homo_results']):
            os.mkdir(cfg.base_directories['base_folder_homo_results'])

        if not os.path.exists(cfg.base_directories['base_folder_homo_models']):
            os.mkdir(cfg.base_directories['base_folder_homo_models'])

        if not os.path.exists(cfg.base_directories['base_folder_homo_groups']):
            os.mkdir(cfg.base_directories['base_folder_homo_groups'])

        if not os.path.exists(cfg.base_directories['base_folder_homo_results_global_models']):
            os.mkdir(cfg.base_directories['base_folder_homo_results_global_models'])

        if not os.path.exists(cfg.base_directories['base_folder_homo_results_clients_models']):
            os.mkdir(cfg.base_directories['base_folder_homo_results_clients_models'])

        cfg.changable_variables['base_folder_results'] = cfg.base_directories['base_folder_homo_results']
        cfg.changable_variables['base_folder_models'] = cfg.base_directories['base_folder_homo_models']
        cfg.changable_variables['base_folder_groups'] = cfg.base_directories['base_folder_homo_groups']
        cfg.changable_variables['base_folder_results_global_models'] = cfg.base_directories[
            'base_folder_homo_results_global_models']
        cfg.changable_variables['base_folder_results_clients_models'] = cfg.base_directories[
            'base_folder_homo_results_clients_models']
        base_folder_clients_data = 'empty for now'

    if dp:
        if not icg:
            fl_accuracy_loss_model_with_dp_without_icg, global_model_favg_with_dp_without_icg = fed_learning.run_fed_avg(
                cfg.data_paths['dataset_path'],
                cfg.data_paths['test_dataset_path'],
                cfg.basic_variables['batch_size'],
                cfg.basic_variables['data_name'],
                cfg.basic_variables['input_shape'],
                cfg.basic_variables['number_of_classes'],
                dp=True,
                icg_bool=False,
                seq=False
            )

            a_file = open(
                f"{cfg.changable_variables['base_folder_results']}/fl_accuracy_loss_model_with_dp_without_icg_customized_clustering.pkl",
                "wb")
            pickle.dump(fl_accuracy_loss_model_with_dp_without_icg, a_file)
            a_file.close()

            global_model_favg_with_dp_without_icg.save(
                f'{cfg.changable_variables["base_folder_models"]}/global_model_favg_with_dp_without_icg_customized_clustering.keras')

            if icg:
                a_file = open(
                    f"{cfg.changable_variables['base_folder_groups']}/fl_groups_with_dp_without_icg_customized_clustering.pkl",
                    "wb")
                pickle.dump(cfg.changable_variables['groups_over_rounds'], a_file)
                a_file.close()

        else:
            fl_accuracy_loss_model_with_icg_dp, global_model_favg_with_icg_dp, cfg.changable_variables[
                'groups_over_rounds'] = fed_learning.run_fed_avg(cfg.data_paths['dataset_path'],
                                                                 cfg.data_paths['test_dataset_path'],
                                                                 cfg.basic_variables['batch_size'],
                                                                 cfg.basic_variables['data_name'],
                                                                 cfg.basic_variables['input_shape'],
                                                                 cfg.basic_variables['number_of_classes'],
                                                                 dp=True,
                                                                 icg_bool=True,
                                                                 seq=False
                                                                 )

            a_file = open(
                f"{cfg.changable_variables['base_folder_results']}/fl_accuracy_loss_model_with_icg_dp_customized_clustering.pkl",
                "wb")
            pickle.dump(fl_accuracy_loss_model_with_icg_dp, a_file)
            a_file.close()

            global_model_favg_with_icg_dp.save(
                f'{cfg.changable_variables["base_folder_models"]}/global_model_favg_with_icg_dp_customized_clustering.keras')

            if icg:
                a_file = open(
                    f"{cfg.changable_variables['base_folder_groups']}/fl_groups_with_icg_dp_customized_clustering.pkl",
                    "wb")
                pickle.dump(cfg.changable_variables['groups_over_rounds'], a_file)
                a_file.close()
    else:
        if not icg:
            fl_accuracy_loss_model_without_dp_icg, global_model_favg_without_dp_icg = fed_learning.run_fed_avg(
                cfg.data_paths['dataset_path'],
                cfg.data_paths['test_dataset_path'],
                cfg.basic_variables['batch_size'],
                cfg.basic_variables['data_name'],
                cfg.basic_variables['input_shape'],
                cfg.basic_variables['number_of_classes'],
                dp=False,
                icg_bool=False,
                seq=False
            )
            a_file = open(
                f"{cfg.changable_variables['base_folder_results']}/fl_accuracy_loss_model_without_dp_icg_customized_clustering.pkl",
                "wb")
            pickle.dump(fl_accuracy_loss_model_without_dp_icg, a_file)
            a_file.close()

            global_model_favg_without_dp_icg.save(
                f'{cfg.changable_variables["base_folder_models"]}/global_model_favg_without_dp_icg_customized_clustering.keras')

            if icg:
                a_file = open(
                    f"{cfg.changable_variables['base_folder_groups']}/fl_groups_without_dp_icg_customized_clustering.pkl",
                    "wb")
                pickle.dump(cfg.changable_variables['groups_over_rounds'], a_file)
                a_file.close()
        else:
            fl_accuracy_loss_model_with_icg_without_dp, global_model_favg_with_icg_without_dp, cfg.changable_variables[
                'groups_over_rounds'] = fed_learning.run_fed_avg(cfg.data_paths['dataset_path'],
                                                                 cfg.data_paths['test_dataset_path'],
                                                                 cfg.basic_variables['batch_size'],
                                                                 cfg.basic_variables['data_name'],
                                                                 cfg.basic_variables['input_shape'],
                                                                 cfg.basic_variables['number_of_classes'],
                                                                 dp=False,
                                                                 icg_bool=True,
                                                                 seq=False)

            a_file = open(
                f"{cfg.changable_variables['base_folder_results']}/fl_accuracy_loss_model_with_icg_without_dp_customized_clustering.pkl",
                "wb")
            pickle.dump(fl_accuracy_loss_model_with_icg_without_dp, a_file)
            a_file.close()

            global_model_favg_with_icg_without_dp.save(
                f'{cfg.changable_variables["base_folder_models"]}/global_model_favg_with_icg_without_dp_customized_clustering.keras')

            if icg:
                a_file = open(
                    f"{cfg.changable_variables['base_folder_groups']}/fl_groups_with_icg_without_dp_customized_clustering.pkl",
                    "wb")
                pickle.dump(cfg.changable_variables['groups_over_rounds'], a_file)
                a_file.close()


if __name__ == '__main__':

    call_fl(dp=cfg.basic_variables['differential_privacy'],
            icg=cfg.basic_variables['clustering'],
            experiment_type=cfg.basic_variables['gen_type'])

    if cfg.basic_variables['plot']:
        cen_global_model_path = open(
            f'{cfg.base_directories["base_folder_centralized_results"]}/centralized_model_tuple.pkl', "rb")
        cen_global_model_tuple = pickle.load(cen_global_model_path)

        fl_accuracy_loss_model_with_icg_dp, fl_accuracy_loss_model_with_dp_without_icg, fl_accuracy_loss_model_with_icg_without_dp, fl_accuracy_loss_model_without_dp_icg = loading_results(
            cfg.changable_variables['base_folder_results'])

        plotting = FLTrainingResultsVisualizer('No title',
                                               ['base_folder_results_global_models'],
                                               cfg.basic_variables['number_of_users']
                                               )
        plotting.extract_global_metrics(
            fl_clients_round_accuracy_with_dp_without_icg=fl_accuracy_loss_model_with_dp_without_icg,
            fl_clients_round_accuracy_without_dp_icg=fl_accuracy_loss_model_without_dp_icg,
            fl_clients_round_accuracy_with_icg_dp=fl_accuracy_loss_model_with_icg_dp,
            fl_clients_round_accuracy_with_icg_without_dp=fl_accuracy_loss_model_with_icg_without_dp,
            cen_global_model_tuple=cen_global_model_tuple)

        if cfg.basic_variables_for_plotting['differential_privacy']:
            if cfg.basic_variables_for_plotting['models_curves'] and not cfg.basic_variables_for_plotting[
                'clients_curves']:
                plotting.set_title('Comprision Between Global Models (DP)')
                visualize_results_for_models(plotting=plotting,
                                             models=cfg.basic_variables_for_plotting['models_curves'],
                                             clients=cfg.basic_variables_for_plotting['clients_curves'],
                                             )

            elif not cfg.basic_variables_for_plotting['models_curves'] and cfg.basic_variables_for_plotting[
                'clients_curves']:
                plotting.set_title('Comprision Between clients Models (DP)')
                visualize_results_for_models(plotting=plotting,
                                             models=cfg.basic_variables_for_plotting['models_curves'],
                                             clients=cfg.basic_variables_for_plotting['clients_curves'],
                                             )
            if cfg.basic_variables_for_plotting['other_metrics']:
                plotting.plot_other_metrics()

        if not cfg.basic_variables_for_plotting['differential_privacy']:
            if cfg.basic_variables_for_plotting['models_curves'] and not cfg.basic_variables_for_plotting[
                'clients_curves']:
                plotting.set_title('Comprision Between Global Models (NON DP)')
                visualize_results_for_models(plotting=plotting,
                                             models=cfg.basic_variables_for_plotting['models_curves'],
                                             clients=cfg.basic_variables_for_plotting['clients_curves'],
                                             )

            elif not cfg.basic_variables_for_plotting['models_curves'] and cfg.basic_variables_for_plotting[
                'clients_curves']:
                plotting.set_title('Comprision Between clients Models (NON DP)')
                visualize_results_for_models(plotting=plotting,
                                             models=cfg.basic_variables_for_plotting['models_curves'],
                                             clients=cfg.basic_variables_for_plotting['clients_curves'],
                                             )
            if cfg.basic_variables_for_plotting['other_metrics']:
                plotting.plot_other_metrics()

        if cfg.basic_variables_for_plotting['clusters_histograms']:
            visualize_histograms_for_clusters(clients_data=cfg.changable_variables['clients_data'],
                                              visualization_title='Clusters Bars Over Rounds',
                                              groups_over_rounds=cfg.changable_variables['groups_over_rounds'],
                                              classes=cfg.changable_variables['classes'])
