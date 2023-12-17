from Codes.Data.InductionForData import InductionForData
from Codes.Models.SimpleMLP import SimpleMLP
import numpy as np


class Client:
    id = 0

    def __init__(self,
                 dataset_path,
                 batch_size,
                 data_name,
                 input_shape,
                 optimizer,
                 loss,
                 metrics,
                 number_of_users,
                 number_of_classes,
                 non_iid_strength_factor,
                 sigma,
                 classes
                 ):
        self.id += 1
        self.number_of_users = number_of_users
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.data_name = data_name
        self.non_iid_strength_factor = non_iid_strength_factor

        self.sigma = sigma
        self.classes = classes
        self.data_induction = InductionForData(self.dataset_path,
                                               self.batch_size,
                                               self.data_name,
                                               self.number_of_users,
                                               self.non_iid_strength_factor,
                                               self.sigma,
                                               self.classes
                                               )
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.number_of_classes = number_of_classes

    def get_user_id(self):
        """
         Get the user id. This is used to check if the user is logged in to the system.
         
         :returns: The user id or None if not logged in to the system ( in this case we don't have a user
        """
        return self.id

    def build_local_model_mnist(self):
        """
         Build and compile MNIST SMLP model. This is a wrapper for SimpleMLP.
         
         :returns: A compiled and built MNIST SMLP model with optimization applied to it. It is used to train the model
        """
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(self.input_shape, self.number_of_classes)
        local_model.compile(loss=self.loss, optimizer='Adam', metrics=self.metrics)
        return local_model

    def build_local_model_daisee(self):
        """
         Build the daisee for the local model. It is called by DAISolver. build_
        """
        pass

    def build_model(self):
        """
         Build model according to data_name. This is a wrapper for build_local_model_mnist or build_local_model_daisee
         
         :returns: a dictionary with keys : model_name : name of the
        """
        if self.data_name == 'MNIST':
            return self.build_local_model_mnist()
        elif self.data_name == 'DAISEE':
            return self.build_local_model_daisee()

    def read_full_daisee_data(self):
        """
         Read data from daisee. This is a wrapper around : py : meth : ` ~data_induction. read_data ` to allow data to be read from a file
         
         :returns: a dataset with the
        """
        dataset = self.data_induction.read_data()
        return dataset

    def read_full_mnist_data(self):
        """
         Reads data from MNIST and returns it. This is a wrapper for data_induction. read_data
         
         :returns: list of clients that were
        """
        clients_data_updated = self.data_induction.read_data()
        return clients_data_updated

    def read_batched_mnist_data(self, clients_data_updated):
        """
         Reads MNIST data from HDF5 and returns a dataset. This is a wrapper around the DataInduction. read_batched_data method which uses the number of classes to read 
         
         :param clients_data_updated: list of clients that have been updated
         :returns: dataset of training data ( numpy array ) and validation data ( numpy array ) for each client ( int
        """
        train_dataset = self.data_induction.read_batched_data(self.number_of_classes,
                                                              clients_data_updated=clients_data_updated)
        return train_dataset

    def read_batched_equal_split_mnist_data(self, clients_data_upda=None):
        """
         Read batched equal MNIST data. This is a wrapper around : py : meth : ` ~data_induction. read_batched_equal_split_mnist_data ` to avoid having to re - read the data in a single call.
         
         :param clients_data_upda: Updatable data that was used to train the model.
         :returns: A list of : py : class : ` ~astropy. table. Table ` objects one for each client
        """
        return self.data_induction.read_batched_equal_split_mnist_data(clients_data_upda=clients_data_upda)

    def read_batched_daisee_data(self, user_id):
        """
         Read batched daisee data. This is a convenience method for calling : py : meth : ` DataInduction. read_batched_data ` on
         
         :param user_id: The id of the user.
         :returns: A list of data in the format returned by : py : meth : ` DataInduction. read
        """
        dataset = self.data_induction.read_batched_data(user_id)
        return dataset

    def read_nor_data(self, magic_value, mu, images, label_list, start, end, gen_type, act):
        """
        Generate data based on the type that is desired as (Heterogeneous, Extreme Heterogeneous, Homogeneous) 
        
        :param magic_value: Value that is used to define the number of samples to be added for each class or for the selected class in case of generating heterogeneous data based on normal distribution
        :param mu: Holds the centralized class for generating heterogeneous data based on normal distribution 
        :param images: List of images to be processed
        :param label_list: List of labels for each image ( 0 - indexed )
        :param start:
        :param end:
        :param gen_type: Type of generation (Heterogeneous, Extreme Heterogeneous, Homogeneous)
        :param act: 
        :returns: User data of shape ( nb_samples n_features ) where n_samples is the number of samples
        """
        user_data = np.array(self.data_induction.produce_data_normal_dis(magic_value,
                                                                         mu,
                                                                         start=start,
                                                                         end=end,
                                                                         gen_type=gen_type,
                                                                         act=act,
                                                                         labels_sums=np.sum(label_list, axis=0)
                                                                         )).astype(int)
        user_data = self.data_induction.process_clients_data(label_list, user_data)
        user_data = self.data_induction.retrieve_data_from_indices(images, label_list, user_data)

        return user_data

    def read_batched_data(self,
                          magic_value=None,
                          mu=None,
                          images=None,
                          label_list=None,
                          user_id=None,
                          clients_data_updated=None,
                          start=0,
                          end=2,
                          act=2,
                          split_type='homo'):
        """
        Read data from NOR. This is a wrapper around read_nor_data to handle MNIST split types
        
        :param magic_value: Value that is used to define the number of samples to be added for each class or for the selected class in case of generating heterogeneous data based on normal distribution
        :param mu: Holds the centralized class for generating heterogeneous data based on normal distribution 
        :param images: List of images to use for missing data ( default None )
        :param label_list: List of labels to use for missing data ( default None )
        :param user_id: User ID to use for missing data ( default None )
        :param clients_data_updated: Data updated by client ( default None
        :param start
        :param end
        :param act
        :param split_type
        """
        if self.data_name == 'MNIST':
            if split_type == 'traditional':
                return self.read_batched_mnist_data(clients_data_updated)
            elif split_type == 'extreme':
                return self.read_nor_data(magic_value, mu, images, label_list, start, end, split_type, act)
            elif split_type == 'hetero':
                return self.read_nor_data(magic_value, mu, images, label_list, start, end, split_type, act)
            elif split_type == 'homo':
                return self.read_nor_data(magic_value, mu, images, label_list, start, end, split_type, act)

        elif self.data_name == 'DAISEE':
            return self.read_batched_daisee_data(user_id)

    def read_full_data(self):
        """
         Read data from the data induction. This is useful for debugging and to avoid having to re - read the data every time it is read.
         
         :returns: A list of numpy arrays one for each data point in the data induction. The list is sorted by the time
        """
        return self.data_induction.read_data()

    def get_local_model_wights(self, model):
        """
         Get the local model weights. This is used to determine whether or not we are going to do an iterative search or not
         
         :param model: The model to be analysed
         :returns: A list of weight values that are in the local model for this iteration or None if there are no
        """
        return model.get_weights()

    def update_local_model_weights(self, model, weights):
        """
         Update the weights of a LocalModel. This is used to update the weights of a LocalModel before it is passed to the model's run method.
         
         :param model: The model to update. Must be of type : py : class : ` pyspark. sql. types. Model `.
         :param weights: The new weights to set. Must be of type : py : class : ` pyspark. sql. types. Weight `.
         :returns: The updated model. Note that the weights are updated in - place and not returned by this method. It is up to the caller to make sure they are in - place
        """
        model = model.set_weights(weights)
        return model

    def get_data_size(self):
        """
         Get the size of the data. This is used to determine how many rows and columns are stored in the data_induction.
         
         :returns: The size of the data in bytes or None if there is no data in the data_induction
        """
        return self.data_induction.get_data_size()

    def prepare_test_data(self, test_dataset_path):
        """
         Prepares test data for use. This is a wrapper around the DataInduction. prepare_test_data method to be able to pass a path to the test dataset and it will return a dictionary of data that can be used to test the data.
         
         :param test_dataset_path: The path to the test dataset.
         :returns: A dictionary of data that can be used to test the data. The keys of the dictionary are the names of the test dataset
        """
        return self.data_induction.prepare_test_data(test_dataset_path)

    def take_batch_of_data(self, data):
        return self.data_induction.take_batch_of_data(data)

