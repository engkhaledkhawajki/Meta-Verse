import numpy as np
import math
import copy
import random


def is_element_in_list_of_lists(element, list_of_lists):

    for sublist in list_of_lists:
        for ccsub in sublist:
            if np.array_equal(ccsub, element):
                return True
    return False


class CustomizedClustering:
    def __init__(self, number_of_clients=10, number_of_clusters=4, clients_data=None):

        self.number_of_clients = number_of_clients
        self.number_of_clusters = number_of_clusters
        self.clients_data = clients_data
        self.clusters = []

    def resolve_clients_data_type(self, clients_data):
        counter = 0
        for client in clients_data:
            for other_client in clients_data:
                lo = np.where(client == 0)[0]
                other_lo = np.where(other_client == 0)[0]
                if len(lo) > 2 and len(other_lo) > 2:
                    if all(item in lo for item in other_lo):
                        counter += 1
                        continue
            break

        groups = [[] for i in range(counter)]
        groups_ids = [[] for i in range(counter)]
        if counter == number_of_users - 1:
            groups = clients_data
        elif counter == 0:
            groups = clients_data


        else:
            for index, client in enumerate(clients_data):
                for j in range(counter):
                    lo = np.where(client == 0)[0]
                    to_sums = np.sum(groups[j], axis=0)
                    indicies = np.where(to_sums == 0)[0]
                    if all(item in lo for item in indicies):
                        continue

                    if len(np.where(client == 0)[0]) > 0:
                        groups[j].append(client)
                        groups_ids[j].append(index)
                        break

                    if len(np.where(client == 0)[0]) == 0:
                        groups[j].append(client)
                        groups_ids[j].append(index)
                        break
        return groups, groups_ids

    def fit(self):

        number_of_clients_in_each_cluster = math.ceil(self.number_of_clients / self.number_of_clusters)
        cluster_data = copy.deepcopy(self.clients_data)
        cluster_data, cluster_ids = self.resolve_clients_data_type(cluster_data)
        clients_data_dict = {}

        for iinndex, cl_data in enumerate(cluster_data):
            clients_data_dict[iinndex] = cl_data

        keys = list(clients_data_dict.keys())
        random.shuffle(keys)
        num_subsets = 2
        if len(cluster_data) <= num_subsets:
            return cluster_ids

        keys_per_subset = len(keys) // 2
        subsets = [keys[i:i + keys_per_subset] for i in range(0, len(keys), keys_per_subset)]
        subsets_data = [{key: clients_data_dict[key] for key in subset} for subset in subsets]

        for bla_bla, cluster_data in enumerate(subsets_data):
            cluster_data_copy = copy.deepcopy(cluster_data)

            for index in range(self.number_of_clusters):

                if len(self.clusters) > index:
                    if len(self.clusters[index]) >= number_of_clients_in_each_cluster:
                        continue

                cluster = []
                max_average = float('-inf')
                max_average_index = None

                if len(cluster_data) > 0 and len(cluster_data) <= index:
                    indin = []

                    for key, arr in cluster_data.items():
                        indin.append(key)
                        cluster.append(key)

                    for keey in indin:
                        if keey in cluster_data:
                            del cluster_data[keey]

                    self.clusters.append(cluster)
                    continue

                elif len(cluster_data) <= 0 and bla_bla >= len(subsets_data):
                    continue

                elif len(cluster_data) <= 0:
                    continue

                else:
                    if bla_bla > 0 and len(self.clusters) > index:
                        client_data = list(cluster_data.keys())[index]
                        cluster.append(client_data)

                        if client_data in cluster_data:
                            del cluster_data[client_data]

                        self.clusters[index].append(client_data)
                        cluster_data_edition = [value for _, value in cluster_data.items()]
                        deviation_matrix = [[0 for _ in range(len(cluster_data[list(cluster_data.keys())[0]]))] for _ in
                                            range(len(cluster_data))]

                        if len(deviation_matrix) < number_of_clients_in_each_cluster:
                            indin = []

                            for key, arr in cluster_data.items():
                                indin.append(key)
                                cluster.append(key)
                                self.clusters[index].append(key)

                            for keey in indin:
                                if keey in cluster_data:
                                    del cluster_data[keey]

                        else:
                            i = len(self.clusters[index])
                            cluster_dev = 0
                            summed_cluster = np.sum([cluster_data_copy[key_to_client] for key_to_client in cluster],
                                                    axis=0)
                            cluster_dev = abs(summed_cluster - max(summed_cluster))

                            for idx, client in enumerate(cluster_data_edition):
                                client_dev = client - cluster_dev
                                deviation_matrix[idx] = client_dev

                            while i < number_of_clients_in_each_cluster:
                                cluster_data_edition = [value for _, value in cluster_data.items()]
                                cluster_dev = 0
                                summed_cluster = np.sum([cluster_data_copy[key_to_client] for key_to_client in cluster],
                                                        axis=0)
                                cluster_dev = abs(summed_cluster - max(summed_cluster))

                                for idx, client in enumerate(cluster_data_edition):
                                    client_dev = client - cluster_dev
                                    deviation_matrix[idx] = client_dev

                                new_array = [abs(v) for v in deviation_matrix]
                                new_array = np.max(new_array, axis=1)
                                no_sort = copy.deepcopy(new_array)
                                new_array = sorted(new_array)
                                key_found = list(cluster_data.keys())[no_sort.tolist().index(new_array[0])]
                                cluster.append(key_found)

                                if key_found in cluster_data:
                                    del cluster_data[key_found]

                                deviation_matrix = np.delete(deviation_matrix, no_sort.tolist().index(new_array[0]),
                                                             axis=0)
                                self.clusters[index].append(key_found)
                                i += 1

                    else:

                        client_data = list(cluster_data.keys())[index]
                        cluster.append(client_data)

                        if client_data in cluster_data:
                            del cluster_data[client_data]

                        cluster_data_edition = [value for _, value in cluster_data.items()]
                        deviation_matrix = [[0 for _ in range(len(cluster_data[list(cluster_data.keys())[0]]))] for _ in
                                            range(len(cluster_data))]

                        if len(deviation_matrix) < number_of_clients_in_each_cluster:
                            indin = []

                            for key, arr in cluster_data.items():
                                indin.append(key)
                                cluster.append(key)

                            for keey in indin:
                                if keey in cluster_data:
                                    del cluster_data[keey]

                        else:
                            i = 1
                            cluster_dev = 0
                            summed_cluster = np.sum([cluster_data_copy[key_to_client] for key_to_client in cluster],
                                                    axis=0)
                            cluster_dev = abs(summed_cluster - max(summed_cluster))

                            for idx, client in enumerate(cluster_data_edition):
                                client_dev = client - cluster_dev
                                deviation_matrix[idx] = client_dev

                            while i < number_of_clients_in_each_cluster:
                                cluster_data_edition = [value for _, value in cluster_data.items()]
                                cluster_dev = 0
                                summed_cluster = np.sum([cluster_data_copy[key_to_client] for key_to_client in cluster],
                                                        axis=0)
                                cluster_dev = abs(summed_cluster - max(summed_cluster))

                                for idx, client in enumerate(cluster_data_edition):
                                    client_dev = client - cluster_dev
                                    deviation_matrix[idx] = client_dev

                                new_array = [abs(v) for v in deviation_matrix]
                                new_array = np.max(new_array, axis=1)
                                no_sort = copy.deepcopy(new_array)
                                new_array = sorted(new_array)
                                key_found = list(cluster_data.keys())[no_sort.tolist().index(new_array[0])]
                                cluster.append(key_found)

                                if key_found in cluster_data:
                                    del cluster_data[key_found]

                                deviation_matrix = np.delete(deviation_matrix, no_sort.tolist().index(new_array[0]),
                                                             axis=0)

                                i += 1

                        self.clusters.append(cluster)

        return self.clusters