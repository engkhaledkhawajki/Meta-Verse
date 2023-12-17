import matplotlib.pyplot as plt
import numpy as np
import copy


class FLTrainingResultsVisualizer:
    def __init__(self, title, path_to_plots, num_clients):
        self.title = title
        self.path_to_plots = path_to_plots
        self.global_accuracy_with_dp_without_icg = []
        self.global_loss_with_dp_without_icg = []
        self.global_recall_with_dp_without_icg = []
        self.global_precision_with_dp_without_icg = []
        self.global_f1score_with_dp_without_icg = []

        self.global_accuracy_without_dp_icg = []
        self.global_loss_without_dp_icg = []
        self.global_recall_without_dp_icg = []
        self.global_precision_without_dp_icg = []
        self.global_f1score_without_dp_icg = []

        self.global_accuracy_with_icg_dp = []
        self.global_loss_with_icg_dp = []
        self.global_recall_with_icg_dp = []
        self.global_precision_with_icg_dp = []
        self.global_f1score_with_icg_dp = []

        self.global_accuracy_with_icg_without_dp = []
        self.global_loss_with_icg_without_dp = []
        self.global_recall_with_icg_without_dp = []
        self.global_precision_with_icg_without_dp = []
        self.global_f1score_with_icg_without_dp = []

        self.cen_model_accuracy = []
        self.cen_model_loss = []

        self.clients_loss_with_dp_without_icg = []
        self.clients_accuracy_with_dp_without_icg = []
        self.clients_rounds_with_dp_without_icg = []

        self.clients_loss_without_dp_icg = []
        self.clients_accuracy_without_dp_icg = []
        self.clients_rounds_without_dp_icg = []

        self.clients_loss_with_icg_dp = []
        self.clients_accuracy_with_icg_dp = []
        self.clients_rounds_with_icg_dp = []

        self.clients_loss_with_icg_without_dp = []
        self.clients_accuracy_with_icg_without_dp = []
        self.clients_rounds_with_icg_without_dp = []

        self.rounds_groups = []
        self.groups = []

        self.num_clients = num_clients

        self.rounds = []

    def set_title(self, new_title: str) -> None:
        self.title = new_title

    def extract_clients_metrics(self, fl_clients_round_accuracy_with_dp_without_icg=None,
                                fl_clients_round_accuracy_without_dp_icg=None,
                                fl_clients_round_accuracy_with_icg_dp=None,
                                fl_clients_round_accuracy_with_icg_without_dp=None):
        """
            Extracts metrics from client.  

            :param fl_clients_round_accuracy_with_dp_without_icg: 
            :param fl_clients_round_accuracy_without_dp_icg: 
            :param fl_clients_round_accuracy_with_icg_dp: 
            :param fl_clients_round_accuracy_with_icg_without_dp: 
        """
        if fl_clients_round_accuracy_with_dp_without_icg is not None:
            for client in range(1, self.num_clients + 1):
                accuracy = []
                losses = []
                rounds_queued = []
                for key, value in fl_clients_round_accuracy_with_dp_without_icg.items():
                    if f'client {client} round' in key:
                        rounds_queued.append(value)
                    elif f'client {client}accuracy' in key:
                        accuracy.append(value)
                    elif f'client {client}loss' in key:
                        losses.append(value)

                self.clients_loss_with_dp_without_icg.append(losses)
                self.clients_accuracy_with_dp_without_icg.append(accuracy)
                self.clients_rounds_with_dp_without_icg.append(rounds_queued)

        if fl_clients_round_accuracy_without_dp_icg is not None:
            for client in range(1, self.num_clients + 1):
                accuracy = []
                losses = []
                rounds_queued = []
                for key, value in fl_clients_round_accuracy_without_dp_icg.items():
                    if f'client {client} round' in key:
                        rounds_queued.append(value)
                    elif f'client {client}accuracy' in key:
                        accuracy.append(value)
                    elif f'client {client}loss' in key:
                        losses.append(value)
                self.clients_loss_without_dp_icg.append(losses)
                self.clients_accuracy_without_dp_icg.append(accuracy)
                self.clients_rounds_without_dp_icg.append(rounds_queued)

        if fl_clients_round_accuracy_with_icg_dp is not None:
            for client in range(1, self.num_clients + 1):
                accuracy = []
                losses = []
                rounds_queued = []
                for key, value in fl_clients_round_accuracy_with_icg_dp.items():
                    if f'client {client} round' in key:
                        rounds_queued.append(value)
                    elif f'client {client}accuracy' in key:
                        accuracy.append(value)
                    elif f'client {client}loss' in key:
                        losses.append(value)

                self.clients_loss_with_icg_dp.append(losses)
                self.clients_accuracy_with_icg_dp.append(accuracy)
                self.clients_rounds_with_icg_dp.append(rounds_queued)

        if fl_clients_round_accuracy_with_icg_without_dp is not None:
            for client in range(1, self.num_clients + 1):
                accuracy = []
                losses = []
                rounds_queued = []
                for key, value in fl_clients_round_accuracy_with_icg_without_dp.items():
                    if f'client {client} round' in key:
                        rounds_queued.append(value)
                    elif f'client {client}accuracy' in key:
                        accuracy.append(value)
                    elif f'client {client}loss' in key:
                        losses.append(value)

                self.clients_loss_with_icg_without_dp.append(losses)
                self.clients_accuracy_with_icg_without_dp.append(accuracy)
                self.clients_rounds_with_icg_without_dp.append(rounds_queued)

    def extract_global_metrics(self, fl_clients_round_accuracy_with_dp_without_icg=None,
                               fl_clients_round_accuracy_without_dp_icg=None,
                               fl_clients_round_accuracy_with_icg_dp=None,
                               fl_clients_round_accuracy_with_icg_without_dp=None,
                               cen_global_model_tuple=None):
        """
            Extracts metrics from Federated models.
            
            :param fl_clients_round_accuracy_with_dp_without_icg:
            :param fl_clients_round_accuracy_without_dp_icg: 
            :param fl_clients_round_accuracy_with_icg_dp: 
            :param fl_clients_round_accuracy_with_icg_without_dp: 
            :param cen_global_model_tuple: 
        """

        if fl_clients_round_accuracy_with_dp_without_icg is not None:
            for key, value in fl_clients_round_accuracy_with_dp_without_icg.items():
                if 'client 1' in key:
                    if len(self.rounds) != 25:
                        if 'round' in key:
                            self.rounds.append(value)
                if 'global model' in key:
                    if 'accuracy' in key:
                        self.global_accuracy_with_dp_without_icg.append(value)
                    if 'loss' in key:
                        self.global_loss_with_dp_without_icg.append(value)
                    if 'precision' in key:
                        self.global_precision_with_dp_without_icg.append(value)
                    if 'recall' in key:
                        self.global_recall_with_dp_without_icg.append(value)
                    if 'f1_score' in key:
                        self.global_f1score_with_dp_without_icg.append(value)

        if fl_clients_round_accuracy_without_dp_icg is not None:
            for key, value in fl_clients_round_accuracy_without_dp_icg.items():
                if 'client 1' in key:
                    if len(self.rounds) != 25:
                        if 'round' in key:
                            self.rounds.append(value)
                if 'global model' in key:
                    if 'accuracy' in key:
                        self.global_accuracy_without_dp_icg.append(value)
                    if 'loss' in key:
                        self.global_loss_without_dp_icg.append(value)
                    if 'precision' in key:
                        self.global_precision_without_dp_icg.append(value)
                    if 'recall' in key:
                        self.global_recall_without_dp_icg.append(value)
                    if 'f1_score' in key:
                        self.global_f1score_without_dp_icg.append(value)

        if fl_clients_round_accuracy_with_icg_dp is not None:
            for key, value in fl_clients_round_accuracy_with_icg_dp.items():
                if 'client 1' in key:
                    if len(self.rounds) != 25:
                        if 'round' in key:
                            self.rounds.append(value)
                if 'global model' in key:
                    if 'accuracy' in key:
                        self.global_accuracy_with_icg_dp.append(value)
                    if 'loss' in key:
                        self.global_loss_with_icg_dp.append(value)
                    if 'precision' in key:
                        self.global_precision_with_icg_dp.append(value)
                    if 'recall' in key:
                        self.global_recall_with_icg_dp.append(value)
                    if 'f1_score' in key:
                        self.global_f1score_with_icg_dp.append(value)

        if fl_clients_round_accuracy_with_icg_without_dp is not None:
            for key, value in fl_clients_round_accuracy_with_icg_without_dp.items():
                if 'client 1' in key:
                    if len(self.rounds) != 25:
                        if 'round' in key:
                            self.rounds.append(value)
                if 'global model' in key:
                    if 'accuracy' in key:
                        self.global_accuracy_with_icg_without_dp.append(value)
                    if 'loss' in key:
                        self.global_loss_with_icg_without_dp.append(value)
                    if 'precision' in key:
                        self.global_precision_with_icg_without_dp.append(value)
                    if 'recall' in key:
                        self.global_recall_with_icg_without_dp.append(value)
                    if 'f1_score' in key:
                        self.global_f1score_with_icg_without_dp.append(value)

        if cen_global_model_tuple is not None:
            for i in range(len(cen_global_model_tuple[0])):
                if len(self.rounds) != 25:
                    self.rounds.append(i)
                self.cen_model_accuracy.append(cen_global_model_tuple[0][i])
                self.cen_model_loss.append(cen_global_model_tuple[1][i])

    def plot_other_metrics(self):
        """
        Plot global model metrics over rounds.

        """
        fig, ax = plt.subplots()
        legends_list = []
        self.rounds = [i for i in range(len(self.rounds))]
        if len(self.global_recall_with_dp_without_icg) > 0:
            ax.plot(self.rounds, self.global_recall_with_dp_without_icg)
            ax.plot(self.rounds, self.global_precision_with_dp_without_icg)
            ax.plot(self.rounds, self.global_f1score_with_dp_without_icg)
            legends_list.append('recall dp|no clustering')
            legends_list.append('precision dp|no clustering')
            legends_list.append('f1 score dp|no clustering')

        if len(self.global_recall_without_dp_icg) > 0:
            ax.plot(self.rounds, self.global_recall_without_dp_icg)
            ax.plot(self.rounds, self.global_precision_without_dp_icg)
            ax.plot(self.rounds, self.global_f1score_without_dp_icg)
            legends_list.append('recall non dp|no clustering')
            legends_list.append('precision non dp|no clustering')
            legends_list.append('f1 score non dp|no clustering')

        if len(self.global_recall_with_icg_dp) > 0:
            ax.plot(self.rounds, self.global_recall_with_icg_dp)
            ax.plot(self.rounds, self.global_precision_with_icg_dp)
            ax.plot(self.rounds, self.global_f1score_with_icg_dp)
            legends_list.append('recall dp|clustering')
            legends_list.append('precision dp|clustering')
            legends_list.append('f1 score dp|clustering')

        if len(self.global_recall_with_icg_without_dp) > 0:
            ax.plot(self.rounds, self.global_recall_with_icg_without_dp)
            ax.plot(self.rounds, self.global_precision_with_icg_without_dp)
            ax.plot(self.rounds, self.global_f1score_with_icg_without_dp)
            legends_list.append('recall non dp|clustering')
            legends_list.append('precision non dp|clustering')
            legends_list.append('f1 score non dp|clustering')

        ax.legend(legends_list)
        ax.set_title(f'{self.title} metrics')
        plt.savefig(f'{self.path_to_plots}{self.title}_metrics.jpg')
        plt.savefig(f'{self.path_to_plots}{self.title}_metrics.svg')

    def plot_loss(self):
        """
        Plot global model loss over rounds.

        """
        fig, ax = plt.subplots()
        legends_list = []
        self.rounds = [i for i in range(len(self.rounds))]
        if len(self.global_loss_with_dp_without_icg) > 0:
            ax.plot(self.rounds, self.global_loss_with_dp_without_icg)
            legends_list.append('dp|no clustering')

        if len(self.global_loss_without_dp_icg) > 0:
            ax.plot(self.rounds, self.global_loss_without_dp_icg)
            legends_list.append('non dp|no clustering')

        if len(self.global_loss_with_icg_dp) > 0:
            ax.plot(self.rounds, self.global_loss_with_icg_dp)
            legends_list.append('dp|clustering')

        if len(self.global_loss_with_icg_without_dp) > 0:
            ax.plot(self.rounds, self.global_loss_with_icg_without_dp)
            legends_list.append('non dp|clustering')

        if len(self.cen_model_loss) > 0:
            ax.plot(self.rounds, self.cen_model_loss)
            legends_list.append('centrlized model')

        ax.legend(legends_list)
        ax.set_title(f'{self.title} Loss')
        plt.savefig(f'{self.path_to_plots}{self.title}_loss.jpg')
        plt.savefig(f'{self.path_to_plots}{self.title}_loss.svg')

    def plot_bars(self, v, relative='clients'):
        """
        Plot histogram for clusters over rounds.

        """
        num_datasets = len(v)
        class_names = [str(index) for index in range(len(v))]
        fig, axs = plt.subplots(num_datasets, figsize=(12, 60))

        plt.subplots_adjust(wspace=0.9)
        for i in range(num_datasets):
            axs[i].bar(class_names, v[i], color='skyblue')
            axs[i].set_xlabel('Class')
            axs[i].set_ylabel('Number of Samples')
            axs[i].set_title(f'{relative} {i}')
            axs[i].set_xticklabels(class_names, rotation=45)

        plt.savefig('clients classes histogram.jpg')
        plt.savefig('clients classes histogram.svg')
        plt.tight_layout()
        plt.show()

    def plot_accuracy(self):
        """
        Plot global model accuracy over rounds.

        """
        fig, ax = plt.subplots()
        self.rounds = [i for i in range(len(self.rounds))]

        legends_list = []
        if len(self.global_accuracy_with_dp_without_icg) > 0:
            ax.plot(self.rounds, self.global_accuracy_with_dp_without_icg)
            legends_list.append('dp|no clustering')

        if len(self.global_accuracy_without_dp_icg) > 0:
            ax.plot(self.rounds, self.global_accuracy_without_dp_icg)
            legends_list.append('non dp|no clustering')

        if len(self.global_accuracy_with_icg_dp) > 0:
            ax.plot(self.rounds, self.global_accuracy_with_icg_dp)
            legends_list.append('dp|clustering')

        if len(self.global_accuracy_with_icg_without_dp) > 0:
            ax.plot(self.rounds, self.global_accuracy_with_icg_without_dp)
            legends_list.append('non dp|clustering')

        if len(self.cen_model_accuracy) > 0:
            ax.plot(self.rounds, self.cen_model_accuracy)
            legends_list.append('centrlized model')

        ax.legend(legends_list)
        ax.set_title(f'{self.title} Accuracy')
        plt.savefig(f'{self.path_to_plots}{self.title}_accuracy.jpg')
        plt.savefig(f'{self.path_to_plots}{self.title}_accuracy.svg')

    def plot_accuracy_clients(self):
        """
        Plot clients models accuracy over rounds.

        """
        legends_list = []

        if len(self.clients_accuracy_with_dp_without_icg) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_with_dp_without_icg), figsize=(25, 40))

        if len(self.clients_accuracy_with_icg_dp) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_with_icg_dp), figsize=(25, 40))

        if len(self.clients_accuracy_with_icg_without_dp) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_with_icg_without_dp), figsize=(25, 40))

        if len(self.clients_accuracy_without_dp_icg) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_without_dp_icg), figsize=(25, 40))

        self.rounds = [i for i in range(len(self.rounds))]

        if len(self.clients_accuracy_without_dp_icg) > 0 or len(self.clients_accuracy_with_icg_without_dp) > 0 or len(
                self.clients_accuracy_with_icg_dp) > 0 or len(self.clients_accuracy_with_dp_without_icg) > 0:
            for idx in range(self.num_clients):
                if len(self.cen_model_accuracy) > 0 and len(self.global_accuracy_without_dp_icg) > 0:
                    axs[idx].plot(self.rounds, self.clients_accuracy_without_dp_icg[idx])
                    axs[idx].plot(self.rounds, self.cen_model_accuracy)
                    legends_list.append(f'client{idx} accuracy')
                    legends_list.append('centralized model')

                if len(self.global_accuracy_with_icg_without_dp) > 0:
                    axs[idx].plot(self.rounds, self.clients_accuracy_with_icg_without_dp[idx])
                    axs[idx].plot(self.rounds, self.global_accuracy_with_icg_without_dp)
                    legends_list.append(f'client{idx} accuracy non dp|clustering')
                    legends_list.append(f'global model non dp|clustering')

                if len(self.global_accuracy_with_icg_dp) > 0:
                    axs[idx].plot(self.rounds, self.clients_accuracy_with_icg_dp[idx])
                    axs[idx].plot(self.rounds, self.global_accuracy_with_icg_dp)
                    legends_list.append(f'client{idx} accuracy dp|clustering')
                    legends_list.append(f'global model dp|clustering')

                if len(self.global_accuracy_without_dp_icg) > 0:
                    axs[idx].plot(self.rounds, self.clients_accuracy_without_dp_icg[idx])
                    axs[idx].plot(self.rounds, self.global_accuracy_without_dp_icg)
                    legends_list.append(f'client{idx} accuracy non dp|no clustering')
                    legends_list.append(f'global model non dp|no clustering')

                if len(self.global_accuracy_with_dp_without_icg) > 0:
                    axs[idx].plot(self.rounds, self.clients_accuracy_with_dp_without_icg[idx])
                    axs[idx].plot(self.rounds, self.global_accuracy_with_dp_without_icg)
                    legends_list.append(f'client{idx} accuracy dp|no clustering')
                    legends_list.append(f'global model dp|no clustering')

                axs[idx].legend(legends_list, loc='center left', bbox_to_anchor=(1, 0.5))
                axs[idx].set_title(f'client {idx} with global model accuracy')

        plt.savefig(f'{self.path_to_plots}{self.title}_accuracy.jpg', bbox_inches="tight")
        plt.savefig(f'{self.path_to_plots}{self.title}_accuracy.svg', bbox_inches="tight")

    def plot_loss_clients(self):
        """
        Plot clients models loss over rounds.

        """
        self.rounds = [i for i in range(len(self.rounds))]
        legends_list = []

        if len(self.clients_accuracy_with_dp_without_icg) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_with_dp_without_icg), figsize=(25, 40))

        if len(self.clients_accuracy_with_icg_dp) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_with_icg_dp), figsize=(25, 40))

        if len(self.clients_accuracy_with_icg_without_dp) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_with_icg_without_dp), figsize=(25, 40))

        if len(self.clients_accuracy_without_dp_icg) > 0:
            fig, axs = plt.subplots(len(self.clients_accuracy_without_dp_icg), figsize=(25, 40))

        if len(self.clients_loss_without_dp_icg) > 0 or len(self.clients_loss_with_icg_without_dp) > 0 or len(
                self.clients_loss_with_icg_dp) > 0 or len(self.clients_loss_with_dp_without_icg) > 0:
            for idx in range(self.num_clients):
                print(idx)
                if len(self.cen_model_loss) > 0 and len(self.global_loss_without_dp_icg) > 0:
                    axs[idx].plot(self.rounds, self.clients_loss_without_dp_icg[idx])
                    axs[idx].plot(self.rounds, self.cen_model_loss)
                    legends_list.append(f'client{idx} accuracy')
                    legends_list.append('centralized model')

                if len(self.global_loss_with_icg_without_dp) > 0:
                    axs[idx].plot(self.rounds, self.clients_loss_with_icg_without_dp[idx])
                    axs[idx].plot(self.rounds, self.global_loss_with_icg_without_dp)
                    legends_list.append(f'client{idx} accuracy non dp|clustering')
                    legends_list.append(f'global model non dp|clustering')

                if len(self.global_loss_with_icg_dp) > 0:
                    axs[idx].plot(self.rounds, self.clients_loss_with_icg_dp[idx])
                    axs[idx].plot(self.rounds, self.global_loss_with_icg_dp)
                    legends_list.append(f'client{idx} accuracy dp|clustering')
                    legends_list.append(f'global model dp|clustering')

                if len(self.global_loss_without_dp_icg) > 0:
                    axs[idx].plot(self.rounds, self.clients_loss_without_dp_icg[idx])
                    axs[idx].plot(self.rounds, self.global_loss_without_dp_icg)
                    legends_list.append(f'client{idx} accuracy non dp|no clustering')
                    legends_list.append(f'global model non dp|no clustering')

                if len(self.global_loss_with_dp_without_icg) > 0:
                    axs[idx].plot(self.rounds, self.clients_loss_with_dp_without_icg[idx])
                    axs[idx].plot(self.rounds, self.global_loss_with_dp_without_icg)
                    legends_list.append(f'client{idx} accuracy dp|no clustering')
                    legends_list.append(f'global model dp|no clustering')

                axs[idx].legend(legends_list, loc='center left', bbox_to_anchor=(1, 0.5))
                axs[idx].set_title(f'client {idx} with global model accuracy')

        plt.savefig(f'{self.path_to_plots}{self.title}_loss.jpg', bbox_inches="tight")
        plt.savefig(f'{self.path_to_plots}{self.title}_loss.svg', bbox_inches="tight")

    def extract_groups_info(self, groups_over_rounds=None):
        """
         Extract information about groups. 

         :param groups_over_rounds: 
        """

        for i in groups_over_rounds.keys():
            if f'groups' in i:
                self.groups.append(groups_over_rounds[i])
            if 'round' in i:
                self.rounds_groups.append(groups_over_rounds[i])

    def plot_clusters_bars(self, class_number, clients_data):
        """
        Plot histogram for clusters over rounds.

        :param class_number: 
        :param clients_data: 
        """
        for _ in clients_data:
            class_names = [str(i) for i in range(len(class_number))]

        for roundd in range(len(self.groups)):
            cclients_groups = copy.deepcopy(self.groups[roundd])

            for i in range(len(cclients_groups)):
                for j in range(len(cclients_groups[i])):
                    cclients_groups[i][j] = clients_data[cclients_groups[i][j]]

            summed_array = []
            for i in range(len(cclients_groups)):
                summed_array.append(np.sum(cclients_groups[i], axis=0))

            num_datasets = len(summed_array)
            fig, axs = plt.subplots(num_datasets, figsize=(12, 20))

            plt.subplots_adjust(wspace=0.9)
            for i in range(num_datasets):
                axs[i].bar(class_names, summed_array[i], color='skyblue')
                axs[i].set_xlabel('Class')
                axs[i].set_ylabel('Number of Samples')
                axs[i].set_title(f'cluster {i}')
                axs[i].set_xticklabels(class_names, rotation=45)

            plt.savefig(f'{self.path_to_plots}clusters classes histogram_{roundd}.jpg')
            plt.savefig(f'{self.path_to_plots}clusters classes histogram_{roundd}.svg')
            plt.tight_layout()
