from sklearn.preprocessing import LabelBinarizer
from tqdm.notebook import tqdm as tq
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import copy
from scipy.stats import norm


class InductionForData:

    def __init__(self, dataset_path,
                 batch_size,
                 data_name,
                 number_of_users,
                 non_iid_strength_factor,
                 sigma,
                 classes
                 ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.data_name = data_name
        self.number_of_users = number_of_users
        self.non_iid_strength_factor = non_iid_strength_factor
        self.sigma = sigma
        self.classes = classes
        self.selected_indices = []

    def process_clients_data(self, label_list, client):
        """
         Process data sent from clients. This is called after the client has been created and is ready to be sent to the client's client_data attribute.
         
         :param label_list: List of labels that have been assigned to the client.
         :param client: Client that is going to be processed by this plugin
        """

        clients_all_labels = []

        for idx, label in enumerate(client):
            client_label_zero = []
            i = 0

            ss = np.where(np.argmax(label_list, axis=1) == idx)[0]

            while i < client[idx]:
                if ss[i] not in self.selected_indices:
                    client_label_zero.append(ss[i])
                    self.selected_indices.append(ss[i])
                    i += 1
                else:
                    i += 1
                    client[idx] += 1
                    continue

            clients_all_labels.append(client_label_zero)

        return clients_all_labels

    def retrieve_data_from_indices(self, images, labels, indices):
        """
         Retrieve data from indices.
         
         :param images: A list of images to be used for retrieval.
         :param labels: A list of labels to be used for retrieval.
         :param indices: A list of indices to be retrieved from the images
        """
        flattened_data_images = []
        flattened_data_labels = []

        for i in range(len(indices)):
            flattened_data_images.append(np.array(images)[indices[i]])
            flattened_data_labels.append(np.array(labels)[indices[i]])

        flattened_data_images = [item for sublist in flattened_data_images for item in sublist]
        flattened_data_labels = [item for sublist in flattened_data_labels for item in sublist]

        dataset = tf.data.Dataset.from_tensor_slices((flattened_data_images, flattened_data_labels))

        return dataset.shuffle(len(flattened_data_labels)).batch(self.batch_size)

    def produce_data_normal_dis(self, magic_value, mu, start=None, end=None, gen_type='hetero', act=2,
                                labels_sums=None):
        """
         Produce data of normal distribution . 
         
        :param magic_value: Value that is used to define the number of samples to be added for each class or for the selected class in case of generating heterogeneous data based on normal distribution
        :param mu: Holds the centralized class for generating heterogeneous data based on normal distribution 
        :param start: 
        :param end:
        :param gen_type: Type of random numbers to generate(Heterogeneous, Extreme Heterogeneous, Homogeneous)
        :param act:
        :param labels_sums: List of labels summed over time
        """
        clients_data = []
        classes = copy.deepcopy(self.classes)
        ranged_values = [0 for _ in range(len(classes))]

        if gen_type == 'hetero':
            print(mu, classes)
            pdf_values = norm.pdf(classes, loc=mu, scale=self.sigma)
            max_pdf = max(pdf_values)
            pdf_values = pdf_values / max_pdf
            print(pdf_values)
            for i in range(magic_value):
                for x, scaled_pdf in zip(classes, pdf_values):
                    ranged_values[int(x)] += scaled_pdf

        if gen_type == 'extreme':
            classes = classes[start:end]
            pdf_values = np.random.uniform(1, 1, len(classes))
            for x, scaled_pdf in zip(classes, pdf_values):
                for _ in range(labels_sums[int(x)] // act):
                    ranged_values[int(x)] += scaled_pdf

        if gen_type == 'homo':
            pdf_values = np.random.uniform(1, 1, len(classes))
            for i in range(magic_value):
                for x, scaled_pdf in zip(classes, pdf_values):
                    ranged_values[int(x)] += scaled_pdf

        clients_data.append(ranged_values)
        return ranged_values

    def batch_data_upd(self, data_shard):
        """Takes in a clients data shard and create a tfds object off it
        args:
            shard: a data, label constituting a client's data shard
            bs:batch size
        return:
            tfds object"""
        data, label = zip(*data_shard)
        dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
        return dataset.shuffle(len(label)).batch(self.batch_size)

    def read_full_daisee_data(self):
        """
         Read data from daisee and return it as tensorflow batched dataset
        """
        labels = []
        image_data = []
        labels_to_number = {'Engagement': 0,
                            'Bordem': 1,
                            'Confusion': 2,
                            'Frustation ': 3,
                            'not_found_label': 4}
        new_path = self.dataset_path
        images = os.listdir(new_path)
        for image in tq(images):
            path_to_image = os.path.join(new_path, image)
            label = image.split('_')
            label = label[-1].split('.')[0]
            try:
                image = np.load(path_to_image, allow_pickle=True)
            except Exception:
                try:
                    image = np.load(path_to_image, allow_pickle=False)
                except Exception as e:
                    print(f"Error loading {image}: {str(e)}")
                    continue

            image_data.append(image)
            labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((list(image_data), list(labels)))

        return dataset.shuffle(len(label)).batch(self.batch_size)

    def read_batched_daisee_data(self, user_id):
        """
         Read batched daisee data. This method is called to read the data for a user.
         
         :param user_id: ID of the user for whom data is
        """
        labels = []
        image_data = []
        labels_to_number = {'Engagement': 0,
                            'Bordem': 1,
                            'Confusion': 2,
                            'Frustation ': 3,
                            'not_found_label': 4}
        new_path = self.dataset_path
        images = os.listdir(new_path)
        for image in tq(images):
            if user_id in image:
                path_to_image = os.path.join(new_path, image)
                label = image.split('_')
                label = label[-1].split('.')[0]
                try:
                    image = np.load(path_to_image, allow_pickle=True)
                except Exception:
                    try:
                        image = np.load(path_to_image, allow_pickle=False)
                    except Exception as e:
                        print(f"Error loading {image}: {str(e)}")
                        continue

                image_data.append(image)
                labels.append(label)

                dataset = tf.data.Dataset.from_tensor_slices((list(image_data), list(labels)))
            else:
                continue

        return dataset.shuffle(len(label)).batch(self.batch_size)

    def read_full_mnist_data(self):
        """
         Read MNIST data and return it as a dictionary. This is a no - op if there is no data
        """
        train_raw = pd.read_csv(self.dataset_path)
        labels = train_raw["label"]
        data = train_raw.drop("label", axis=1)
        images = []

        for i in range(len(data.values)):
            images.append(data.values[i].reshape(28, 28).flatten())

        lb = LabelBinarizer()
        label_list = lb.fit_transform(labels)

        clients_data_upda = []
        for i in range(len(images)):
            clients_data_upda.append((images[i], label_list[i]))
        return clients_data_upda

    def read_batched_mnist_data(self, clients_data_updated, NUM_CLASSES):
        """
         Read MNIST data and split it to clients with a level of heterogenity.
         
         :param clients_data_updated: A list of client ids to update
         :param NUM_CLASSES: The number of classes to
        """
        # NUM_CLASSES = len(np.unique(np.array(y_train)))
        clients_batched_upd = dict()

        clients_data = sorted(clients_data_updated, key=lambda x: int(np.argmax(x[1])))

        # Then, we need to separate the samples by creating a dictionary of 10 different keys (0 -> 9) which are the labels.
        # Each key has a value which is a list of all samples belong to the label that the key represents.
        sorted_by_label = {str(k): [] for k in range(10)}
        for sample in clients_data:
            sorted_by_label[str(int(np.argmax(sample[1])))].append(sample)

        # This dictionary is going to hold the dataset separated by clients.
        # The dictionary will be filled in the for loop below where we distribute the samples between the clients.
        clients_datasets_dict = {k: [] for k in range(self.number_of_users)}

        # The following dictionary contains 10 different keys (0 -> 9) which are the labels.
        # Each value is a list of size equals to self.number_of_users.
        # Each list contains random picks along a range of "label_list" length (indices in ascending order). Some of theses indices will be dropped depening on the non_iid_strength_factor as we will see in the loop below.
        # This means that each client "i" will have chunk of samples from label_list[i-1] to label_list[i] for each label.
        lables_indices = {k: sorted(random.sample(range(len(label_list)), (self.number_of_users - 1))) for
                          label, label_list, k in
                          zip(sorted_by_label.keys(), sorted_by_label.values(), range(NUM_CLASSES))}

        # The dictionary here is meant to track the next available label index for the next client.
        index_tracker = {label: 0 for label, _ in sorted_by_label.items()}

        # Starting the distributing process by first looping over the clients.
        for i in range(self.number_of_users):
            # client_execluded_classes holds the labels that the i-th client will be deprived from
            client_execluded_classes = random.sample(range(NUM_CLASSES), self.non_iid_strength_factor)
            for label, label_list in sorted_by_label.items():
                # If the outer loop is not at the last client:
                if i != (self.number_of_users - 1):
                    # Pop the next index whatever.
                    label_index = lables_indices[int(label)].pop(0)
                    # If the i-th client can't have the current label in the inner loop, continue.
                    if int(label) in client_execluded_classes:
                        continue
                    # If the outer loop is at the last client, give me the last index.
                    else:
                        label_index = len(label_list)
                # Assign the next chunck to the i-th client.
                clients_datasets_dict[i].extend(label_list[index_tracker[label]:label_index])
                random.shuffle(clients_datasets_dict[i])
                index_tracker[label] = label_index

        for (client_name, data) in tq(clients_datasets_dict.items()):
            clients_batched_upd[client_name] = self.batch_data_upd(data, NUM_CLASSES)

        return clients_batched_upd

    def read_batched_equal_split_mnist_data(self, clients_data_upda=None):
        """
        Read MNIST data for equal split. 

        :param clients_data_upda: Upda data from client
        """

        llabels = [lab[1] for lab in clients_data_upda]
        ddata = [lab[0] for lab in clients_data_upda]
        new_order_client = {}
        classes = np.unique(np.argmax(llabels, axis=1))
        first_classes = [cl for cl in classes[:(len(classes) // 2)]]
        second_classes = [cl for cl in classes[(len(classes) // 2):]]

        taken_classes = {label: -1 for label in classes}
        for i in range(self.number_of_users):
            dataa2 = []
            labels2 = []
            for j, q in zip(range(len(first_classes)), range(len(second_classes))):
                if taken_classes[first_classes[j]] == -1 and taken_classes[second_classes[q]] == -1:
                    indices = np.where(first_classes[j] == np.argmax(llabels, axis=1))[0]
                    indices2 = np.where(second_classes[q] == np.argmax(llabels, axis=1))[0]

                    dataa2.append([ddata[im] for im in indices])
                    dataa2.append([ddata[dm] for dm in indices2])
                    labels2.append([llabels[im] for im in indices])
                    labels2.append([llabels[dm] for dm in indices2])
                    dataset = tf.data.Dataset.from_tensor_slices(
                        (np.concatenate((np.array(dataa2[0]), np.array(dataa2[1])), axis=0),
                         np.concatenate((np.array(labels2[0]), np.array(labels2[1])), axis=0))).batch(self.batch_size)

                    new_order_client[i] = dataset
                    taken_classes[j] = 0
                    taken_classes[q] = 0
                    break
                else:
                    continue
        return new_order_client

    def get_mnist_image(self, X_test, y_test):
        """
         Returns MNIST image.

         :param X_test: numpy array of shape ( n_samples n_features )
         :param y_test: numpy array of shape ( n_samples
        """
        rand_idx = random.randrange(len(X_test))
        random_image = X_test[rand_idx]
        random_label = y_test[rand_idx]
        img = random_image.reshape(28, 28)
        return img, random_label

    def get_daisee_image(self, test_data_path):
        """
         Get daisee image. 

         :param test_data_path: Path to the test data
        """
        labels_to_number = {'Engagement': 0,
                            'Bordem': 1,
                            'Confusion': 2,
                            'Frustation ': 3,
                            'not_found_label': 4}
        new_path = test_data_path
        images = os.listdir(new_path)
        img = np.random.choice(images)
        path_to_image = os.path.join(new_path, img)
        label = img.split('_')
        label = label[-1].split('.')[0]
        try:
            image = np.load(path_to_image, allow_pickle=True)
        except Exception:
            try:
                image = np.load(path_to_image, allow_pickle=False)
            except Exception as e:
                print(f"Error loading {image}: {str(e)}")
        return image, label

    def read_data(self):
        """
         Read and return data. This is wrapper function to generate data based on it's name (MNIST, DAISEE)
        """
        if self.data_name == 'DAISEE':
            return self.read_full_daisee_data()
        elif self.data_name == 'MNIST':
            return self.read_full_mnist_data()

    def read_batched_data(self, NUM_CLASSES, user_id=None, clients_data_updated=None, ):
        """
         Read and return data. This is wrapper function to generate data based on it's name (MNIST, DAISEE)
         
         :param NUM_CLASSES: The number of classes to read.
         :param user_id: The user_id of the user who made the request.
         :param clients_data_updated: If not None the list of clients that have been updated
        """
        if self.data_name == 'DAISEE':
            return self.read_batched_daisee_data(user_id)

        elif self.data_name == 'MNIST':
            return self.read_batched_mnist_data(clients_data_updated, NUM_CLASSES)

    def get_daisee_dataset_size(self):
        new_path = self.dataset_path
        images = os.listdir(new_path)
        return len(images)

    def get_mnist_dataset_size(self):
        train_raw = pd.read_csv(self.dataset_path)
        return len(train_raw)

    def get_data_size(self):
        if self.data_name == 'DAISEE':
            return self.get_daisee_dataset_size()
        elif self.data_name == 'MNIST':
            return self.get_mnist_dataset_size()

    def prepare_test_data(self, test_dataset_path):
        train_raw = pd.read_csv(test_dataset_path)
        labels = train_raw["label"]
        data = train_raw.drop("label", axis=1)
        images = []

        for i in range(len(data.values)):
            images.append(data.values[i].reshape(28, 28).flatten())

        lb = LabelBinarizer()
        label_list = lb.fit_transform(labels)

        test_dataset = tf.data.Dataset.from_tensor_slices((images, label_list)).batch(self.batch_size)

        return test_dataset

    def take_batch_of_data(self, data, percentage_to_train_on=0.1):
        remain_data = []
        X_train = []
        y_train = []
        if isinstance(data, tf.data.Dataset):
            for la in list(data):
                for data in la[0]:
                    X_train.append(data.numpy())
                for label in la[1]:
                    y_train.append(label.numpy())

        #                 X_train = [d[0] for d in data.as_numpy_iterator()]
        #                 y_train = [d[1] for d in data.as_numpy_iterator()]

        else:
            X_train = [da[0] for da in data]
            y_train = [da[1] for da in data]

        boundary = int(len(X_train) * percentage_to_train_on)
        new_X_train = X_train[:boundary]
        new_y_train = y_train[:boundary]

        remain_X_train = X_train[boundary:]
        remain_y_train = y_train[boundary:]

        dataset = tf.data.Dataset.from_tensor_slices((new_X_train, new_y_train)).batch(self.batch_size)
        remian_dataset = tf.data.Dataset.from_tensor_slices((remain_X_train, remain_y_train)).batch(self.batch_size)

        return dataset, remian_dataset
