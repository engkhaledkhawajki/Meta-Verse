import numpy as np
import tensorflow as tf

data_paths = {
    'dataset_path': "C:\\Users\lenovo\Downloads\Compressed\mnist_train.csv",
    'test_dataset_path':"C:\\Users\lenovo\Downloads\Compressed\mnist_test.csv"
}

base_directories = {

    'base_folder_centralized_results':'C:/Users/lenovo/Desktop/meta_verse/centralized/results',
    'base_folder_centralized_models':'C:/Users/lenovo/Desktop/meta_verse/centralized/models',
    
    'base_folder_extreme_results':'C:/Users/lenovo/Desktop/meta_verse/extreme/results/',
    'base_folder_extreme_models':'C:/Users/lenovo/Desktop/meta_verse/extreme/models/',
    'base_folder_extreme_groups':'C:/Users/lenovo/Desktop/meta_verse/extreme/groups/global_models/',
    'base_folder_extreme_clients_data':'C:/Users/lenovo/Desktop/meta_verse/extreme/groups/clients_data/',
    'base_folder_extreme_results_global_models':'C:/Users/lenovo/Desktop/meta_verse/extreme/results/global/',
    'base_folder_extreme_results_clients_models':'C:/Users/lenovo/Desktop/meta_verse/extreme/results/clients/',

    'base_folder_hetero_results':'C:/Users/lenovo/Desktop/meta_verse/hetero/results/',
    'base_folder_hetero_models':'C:/Users/lenovo/Desktop/meta_verse/hetero/models/',
    'base_folder_hetero_groups':'C:/Users/lenovo/Desktop/meta_verse/hetero/groups/global_models/',
    'base_folder_hetero_clients_data':'C:/Users/lenovo/Desktop/meta_verse/hetero/groups/clients_data/',
    'base_folder_hetero_results_global_models':'C:/Users/lenovo/Desktop/meta_verse/hetero/results/global/',
    'base_folder_hetero_results_clients_models':'C:/Users/lenovo/Desktop/meta_verse/hetero/results/clients/',

    'base_folder_homo_results':'C:/Users/lenovo/Desktop/meta_verse/homo/results/',
    'base_folder_homo_models':'C:/Users/lenovo/Desktop/meta_verse/homo/models/',
    'base_folder_homo_groups':'C:/Users/lenovo/Desktop/meta_verse/homo/groups/global_models/',
    'base_folder_homo_clients_data':'C:/Users/lenovo/Desktop/meta_verse/homo/groups/clients_data/',

    'base_folder_homo_results_global_models':'C:/Users/lenovo/Desktop/meta_verse/homo/results/global/',
    'base_folder_homo_results_clients_models':'C:/Users/lenovo/Desktop/meta_verse/homo/results/clients/',
}

basic_variables = {
    'data_name':'MNIST',
    'metrics':['accuracy'],
    'loss':tf.keras.losses.CategoricalCrossentropy(),
    'input_shape':784,
    'optimizer':tf.keras.optimizers.Adam(),
    'client_batched':{},
    'clients':{},
    'number_of_classes':10,
    'clipping_norm':10,
    'epsilon':10,
    'comms_round':25,
    'USERS_PER_ROUND':10,
    'learning_rate':0.001,
    'momentum':0.9,
    'number_of_users':10,
    'batch_size':32,
    'non_iid_strength_factor':3,
    'sigma':2,
    'magic_value':500,
    'gen_type':'extreme',
    'the_allowed_hete':2,
    'plot':False,
    'differential_privacy':False,
    'clustering':True,
}

basic_variables_for_plotting ={
    'differential_privacy':False,
    'clustering':True,
    'models_curves':True,
    'clients_curves':False,
    'clusters_histograms':True,
    'other_metrics':True

}

changable_variables = {
    'classes' : None,
    'clients_data' : None,
    'base_folder_results_global_models' : ' ',
    'base_folder_results_clients_models' : ' ',
    'base_folder_groups' : ' ',
    'base_folder_results' : ' ',
    'base_folder_models' : ' ',
    'base_folder_clients_data' : ' ',
    'groups_over_rounds' : None,
}