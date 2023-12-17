import tensorflow as tf
from Codes.Differential_privacy.Diff_privacy import DP
from Codes.Client_Server.Client import Client
from sklearn.metrics import precision_score, recall_score, f1_score
from Codes.Inter_Clustering_Grouping.ICG import CustomizedClustering
import keras.backend as K
import numpy as np
from typing import List, Tuple
import copy


class FL:
    def __init__(self, number_of_clients: int, comm_rounds: int, metrics: List[str], loss_function: str,
                 optimizer: str, server, clipping_norm: float, epsilon: float, user_per_round: int,
                 learning_rate: float, momentum: float, non_iid_strength_factor, sigma, classes, magic_value) -> None:
        self.comm_rounds = comm_rounds
        self.metrics = metrics
        self.clipping_norm = clipping_norm
        self.epsilon = epsilon
        self.user_per_round = user_per_round
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.server = server
        self.number_of_clients = number_of_clients
        self.non_iid_strength_factor = non_iid_strength_factor
        self.sigma = sigma
        self.classes = classes
        self.magic_value = magic_value
        self.data_size = 0
        self.def_privacy = None
        self.clients = {}
        self.clients_batched = {}
        self.test_data = []
        self.global_model_favg = None
        self.data_size = 0
        self.icg_instance = None
        self.beta = 2
        self.alpha = 0.5
        self.clients_groups = None

    def growth_function(self, func_type, current_round: int, num_of_groups=2, the_order='increase') -> float:
        """
         Produce the cluster number based on function type user will choose (increase, monotic, fixed value) and the iteration round
         
         :param func_type: Type of the growth function.
         :param current_round: Iteration round
         :param num_of_groups: Number of groups to be used in the growth function.
         :param the_order: Order of the growth function which is used with the function type (increase).
         :returns: The value of the growth function in the given round
        """
        if func_type == 'monotic':
            return self.beta * (self.alpha * np.log(current_round) + 1)
        elif func_type == 'fixed':
            return 4
        else:
            if the_order == 'increase':
                num_of_groups += 1
                return num_of_groups
            else:
                num_of_groups -= 1
                return num_of_groups

    def prepare_fl(self,
                   dataset_path: str,
                   test_dataset_path: str,
                   batch_size: int,
                   data_name: str,
                   input_shape: Tuple[int, int],
                   number_of_classes: int,
                   gen_type: str,
                   the_allowed_hete: int
                   ) -> dict:

        """
        Prepares the data and the clients to be used with the Federated System 

        :param dataset_path: Path to the dataset to be used for training.
        :param test_dataset_path: Path to the test dataset to be used for testing.
        :param batch_size: Batch size to use for training.
        :param data_name: Name of the data to be used for training.
        :param input_shape: Shape of the input to the dataset.
        :param number_of_classes: Number of classes in the dataset.
        :param gen_type: Type of the data generation to be used.
        :param the_allowed_hete: Maximum number of heterogenity for class to be considered this is used with (extreme heterogeneity).
        :returns: A ready to use FluGrain object that can be used to train and test
        """
        start = 0
        end = 2

        act = ((self.number_of_clients * the_allowed_hete) // 10)
        magic_value = 500

        client = Client(dataset_path,
                        batch_size,
                        data_name,
                        input_shape,
                        self.optimizer,
                        self.loss_function,
                        self.metrics,
                        self.number_of_clients,
                        number_of_classes,
                        self.non_iid_strength_factor,
                        self.sigma,
                        self.classes)
        clients_data_updated = client.read_full_data()
        self.test_data = client.prepare_test_data(test_dataset_path)
        copied_array = copy.deepcopy(self.classes)

        label_list = [label[1] for label in clients_data_updated]
        images = [label[0] for label in clients_data_updated]
        q = 0
        for i in range(1, self.number_of_clients + 1):
            if gen_type == 'extreme':
                magic_value = np.sum(label_list, axis=0)[q] // act
            mu = np.random.randint(0, len(copied_array))
            self.clients[i] = Client(dataset_path,
                                     batch_size,
                                     data_name,
                                     input_shape,
                                     self.optimizer,
                                     self.loss_function,
                                     self.metrics,
                                     self.number_of_clients,
                                     number_of_classes,
                                     self.non_iid_strength_factor,
                                     self.sigma,
                                     self.classes)

            self.clients_batched[i] = self.clients[i].read_batched_data(magic_value=magic_value,
                                                                        mu=copied_array[mu],
                                                                        images=images,
                                                                        label_list=label_list,
                                                                        start=start,
                                                                        end=end,
                                                                        act=act,
                                                                        split_type=gen_type)
            if i % act == 0:
                start = end
                end = end + the_allowed_hete
                q = q + 1
        self.data_size = client.get_data_size()
        self.def_privacy = DP(self.number_of_clients,
                              self.clipping_norm,
                              self.epsilon,
                              self.comm_rounds,
                              self.user_per_round,
                              self.data_size,
                              self.learning_rate,
                              self.momentum)
        return self.clients_batched

    def weight_scaling_factor(self, clients_batched, client_id: int) -> float:
        """
         Weightscalling factor for batch_in. The weightscalling factor is the ratio of the number of local and global clients that have at least one sample in the batch.

         :param clients_batched: A dictionary of client_ids as keys and a list of numpy arrays as values.
         :param client_id: The id of the client to calculate the weightscalling factor for.
         :returns: The weightscalling factor for the batch_in with id client_id as keys and a float
        """
        client_names = list(clients_batched.keys())
        bs = list(clients_batched[client_id])[0][0].shape[0]
        global_count = sum([tf.data.experimental.cardinality(clients_batched[client_name]).numpy() for client_name in
                            client_names]) * bs
        local_count = tf.data.experimental.cardinality(clients_batched[client_id]).numpy() * bs
        return local_count / global_count

    def scale_model_weights(self, weight: List[np.ndarray], scalar: float) -> List[np.ndarray]:
        """
         Scale weights by a scalar. This is used to scale the weights when training a neural network.

         :param weight: List of weights in the form [ x y z ]
         :param scalar: Scalar to scale the weights by. Default is 1.
         :returns: List of weights scaled by scalar ( same length as weight ). Note that the order of weights is preserved
        """
        weight_final = []
        steps = len(weight)
        # Add a weight to the final weight
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final

    def sum_scaled_weights(self, scaled_weight_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
         Sums scaled weights across all layers. This is useful for computing the mean of a list of scaled weights across all layers.

         :param scaled_weight_list: List of lists of scaled weights. Each list is a list of layer weights. Each layer weight is a 2D NumPy array where the first dimension is the batch size
         :returns: List of averaged weights across all layers. The length of the list is the same as the length of the scaled
        """
        avg_grad = list()
        # Computes average gradient of all weights.
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad

    def run_fed_avg(self,
                    dataset_path: str,
                    test_dataset_path: str,
                    batch_size: int,
                    data_name: str,
                    input_shape: Tuple[int, int],
                    number_of_classes: int,
                    dp: bool = False,
                    icg_bool: bool = False,
                    seq: bool = False,
                    metric_used: str = 'accuracy',
                    loss_used: str = 'loss',
                    growth_function_type='monotic'
                    ):
        """
            Runs federated average system. It is used to calculate the performance of the model and return the results in a dictionary
            
            :param dataset_path: Path to the dataset to be used for training
            :param test_dataset_path: Path to the test dataset to be used for testing
            :param batch_size: Number of examples in the dataset ( default : 10 )
            :param data_name: Name of the data to be used for training
            :param input_shape: Shape of the input data ( default : tuple ( int int ))
            :param number_of_classes: Number of classes in the dataset ( default : int ( 10 ))
            :param dp: Boolean if True use DP ( default : False )
            :param icg_bool: Boolean if True use clustering ( default : False )
            :param seq: Boolean if True use seqential training ( default : False )
            :param metric_used: String metric used ( default :'accuracy')
            :param loss_used: String loss used to determine loss ( default :'loss')
            :param growth_function_type: String growth function type ( default : monotic )
            :returns: Dictionary with keys : train_results : List of training
        """

        clipping_norm = self.def_privacy.get_clipping_norm()
        c = self.def_privacy.get_c_value()
        epsilon = self.def_privacy.get_epsilon()
        fl_accuracy_loss_model = {}
        groups_over_rounds = {}
        test_data_data = []
        test_data_label = []
        self.global_model_favg = self.server.build_model()
        print('Training the servers model on 10% of the data')
        data_used_to_prepare_global_model, self.test_data = self.clients[1].take_batch_of_data(self.test_data)
        self.global_model_favg = self.server.compile_fit_model(self.global_model_favg,
                                                               data_used_to_prepare_global_model,
                                                               self.metrics,
                                                               self.loss_function,
                                                               )

        data_used_to_prepare_local_models, self.test_data = self.clients[1].take_batch_of_data(self.test_data)

        global_model_favg_score = self.global_model_favg.evaluate(self.test_data)

        print(global_model_favg_score)
        print('done\n\n\n')

        if icg_bool:
            client_names = list(self.clients_batched.keys())
            for comm_round in range(1, self.comm_rounds + 1):
                global_weights = self.global_model_favg.get_weights()

                print(f"\t__Round__:{comm_round}")
                scaled_local_weight_list = list()

                number_of_groups = self.growth_function(growth_function_type, comm_round)
                keys = list(self.clients_batched.keys())
                V = []

                for key in keys:
                    label_for_each_user = []
                    for la in list(self.clients_batched[key]):
                        for label in la[1]:
                            label_for_each_user.append(label.numpy())
                    V.append(np.sum([label_nes for label_nes in label_for_each_user], axis=0))
                V = np.array(V)
                print('number of groups', number_of_groups)
                clustering = CustomizedClustering(number_of_clients=self.number_of_clients,
                                                  number_of_clusters=int(number_of_groups),
                                                  clients_data=V)
                self.clients_groups = clustering.fit()

                scaled_cluster_weight_list = list()
                groups_over_rounds[f'groups {number_of_groups}'] = self.clients_groups
                groups_over_rounds[f'round'] = comm_round
                for group_id, group in enumerate(self.clients_groups):

                    array_of_avg_in_cluster = np.sum([label_nes for label_nes in label_for_each_user], axis=0) / len(
                        group)

                    cluster_model = self.clients[group_id + 1].build_model()
                    cluster_model.set_weights(global_weights)

                    data_size_for_cluster = 0
                    global_count = 0
                    group_scalar = 0
                    scalars = []
                    summerizer = 0

                    for idx in group:
                        his = cluster_model.fit(self.clients_batched[idx + 1], epochs=1, verbose=1)
                        his = cluster_model.fit(data_used_to_prepare_local_models, epochs=1, verbose=1)

                        cluster_score = cluster_model.evaluate(self.test_data)

                        fl_accuracy_loss_model[f'client {idx + 1}' + metric_used + str(comm_round)] = his.history[
                            metric_used]
                        fl_accuracy_loss_model[f'client {idx + 1}' + loss_used + str(comm_round)] = his.history[
                            loss_used]
                        fl_accuracy_loss_model[f'client {idx + 1}' + ' round ' + str(comm_round)] = comm_round
                        data_size_for_client = list(self.clients_batched[idx + 1])[0][0].shape[0] * len(
                            list(self.clients_batched[idx + 1]))

                        data_size_for_cluster += data_size_for_client
                        cluster_weights = cluster_model.get_weights()

                        if dp:
                            print('started the training with DP')
                            sensitivity = 2 * (clipping_norm) / (self.data_size / self.number_of_clients)
                            stddev = (c * sensitivity * (comm_round + 1)) / epsilon
                            new_cluster_weights = []

                            for index, weight in enumerate(cluster_weights):
                                cluster_weights[index] = self.def_privacy.clip_weights(cluster_weights[index])
                                stddev = tf.cast(stddev, cluster_weights[index].dtype)
                                noise = tf.random.normal(shape=tf.shape(cluster_weights[index]), mean=0.0,
                                                         stddev=stddev)
                                cluster_weights[index] = cluster_weights[index] + noise


                    global_count = sum(
                        [tf.data.experimental.cardinality(self.clients_batched[client_name]).numpy() for client_name in
                         list(self.clients_batched.keys())]) * batch_size

                    print(data_size_for_cluster, global_count)
                    scaling_factor = data_size_for_cluster / global_count
                    scaled_weights = self.scale_model_weights(cluster_weights, scaling_factor)
                    scaled_cluster_weight_list.append(scaled_weights)
                    K.clear_session()
                    average_weights = self.sum_scaled_weights(scaled_cluster_weight_list)

                self.global_model_favg.set_weights(average_weights)

                self.global_model_favg.compile(loss=self.loss_function,
                                               optimizer='Adam',
                                               metrics=self.metrics)

                test_data_data = list(self.test_data)[0][0].numpy()
                test_data_label = list(self.test_data)[0][1].numpy()
                prediction = self.global_model_favg.predict(test_data_data)
                precision = precision_score([np.argmax(i) for i in test_data_label], [np.argmax(j) for j in prediction],
                                            average='macro')
                recall = recall_score([np.argmax(i) for i in test_data_label], [np.argmax(j) for j in prediction],
                                      average='macro')
                f1 = f1_score([np.argmax(i) for i in test_data_label], [np.argmax(j) for j in prediction],
                              average='macro')

                score = self.global_model_favg.evaluate(self.test_data)
                fl_accuracy_loss_model['global model' + ' accuracy ' + str(comm_round)] = score[1]
                fl_accuracy_loss_model['global model' + ' loss ' + str(comm_round)] = score[0]
                fl_accuracy_loss_model['global model' + ' precision ' + str(comm_round)] = precision
                fl_accuracy_loss_model['global model' + ' recall ' + str(comm_round)] = recall
                fl_accuracy_loss_model['global model' + ' f1_score ' + str(comm_round)] = f1

            return fl_accuracy_loss_model, self.global_model_favg, groups_over_rounds

        else:
            if seq:
                for comm_round in range(self.comm_rounds):
                    print(f"\t__Round__:{comm_round}")
                    self.global_model_favg.compile(loss=self.loss_function,
                                                   optimizer='Adam',
                                                   metrics=self.metrics)
                    for idx in range(1, self.number_of_clients + 1):
                        his = self.global_model_favg.fit(self.clients_batched[idx], epochs=1, verbose=1)
                        fl_accuracy_loss_model[f'client {idx}' + metric_used + str(comm_round)] = his.history[
                            metric_used]
                        fl_accuracy_loss_model[f'client {idx}' + loss_used + str(comm_round)] = his.history[loss_used]
                        fl_accuracy_loss_model[f'client {idx}' + ' round ' + str(comm_round)] = comm_round

                    score = self.global_model_favg.evaluate(self.test_data)
                    fl_accuracy_loss_model['global model' + ' accuracy ' + str(comm_round)] = score[1]
                    fl_accuracy_loss_model['global model' + ' loss ' + str(comm_round)] = score[0]
                return fl_accuracy_loss_model, self.global_model_favg
            else:
                for comm_round in range(self.comm_rounds):
                    print(f"\t__Round__:{comm_round}")
                    global_weights = self.global_model_favg.get_weights()

                    scaled_local_weight_list = list()

                    for idx in range(1, self.number_of_clients + 1):
                        local_model = self.clients[idx].build_model()
                        local_model.set_weights(global_weights)
                        his = local_model.fit(self.clients_batched[idx], epochs=1, verbose=1)

                        fl_accuracy_loss_model[f'client {idx}' + metric_used + str(comm_round)] = his.history[
                            metric_used]
                        fl_accuracy_loss_model[f'client {idx}' + loss_used + str(comm_round)] = his.history[loss_used]
                        fl_accuracy_loss_model[f'client {idx}' + ' round ' + str(comm_round)] = comm_round
                        local_weights = local_model.get_weights()

                        if dp:
                            print('started the training with DP')
                            sensitivity = 2 * (clipping_norm) / (self.data_size / self.number_of_clients)
                            stddev = (c * sensitivity * (comm_round + 1)) / epsilon
                            new_local_weights = []

                            for index, weight in enumerate(local_weights):
                                local_weights[index] = self.def_privacy.clip_weights(local_weights[index])
                                stddev = tf.cast(stddev, local_weights[index].dtype)
                                noise = tf.random.normal(shape=tf.shape(local_weights[index]), mean=0.0, stddev=stddev)
                                local_weights[index] = local_weights[index] + noise

                        scaling_factor = self.weight_scaling_factor(self.clients_batched, idx)
                        scaled_weights = self.scale_model_weights(local_weights, scaling_factor)
                        scaled_local_weight_list.append(scaled_weights)
                        K.clear_session()
                        average_weights = self.sum_scaled_weights(scaled_local_weight_list)

                    self.global_model_favg.set_weights(average_weights)
                    self.global_model_favg.compile(loss=self.loss_function,
                                                   optimizer='Adam',
                                                   metrics=self.metrics)

                    test_data_data = list(self.test_data)[0][0].numpy()
                    test_data_label = list(self.test_data)[0][1].numpy()
                    prediction = self.global_model_favg.predict(test_data_data)
                    precision = precision_score([np.argmax(i) for i in test_data_label],
                                                [np.argmax(j) for j in prediction], average='macro')
                    recall = recall_score([np.argmax(i) for i in test_data_label], [np.argmax(j) for j in prediction],
                                          average='macro')
                    f1 = f1_score([np.argmax(i) for i in test_data_label], [np.argmax(j) for j in prediction],
                                  average='macro')

                    score = self.global_model_favg.evaluate(self.test_data)
                    fl_accuracy_loss_model['global model' + ' accuracy ' + str(comm_round)] = score[1]
                    fl_accuracy_loss_model['global model' + ' loss ' + str(comm_round)] = score[0]
                    fl_accuracy_loss_model['global model' + ' precision ' + str(comm_round)] = precision
                    fl_accuracy_loss_model['global model' + ' recall ' + str(comm_round)] = recall
                    fl_accuracy_loss_model['global model' + ' f1_score ' + str(comm_round)] = f1

                return fl_accuracy_loss_model, self.global_model_favg
