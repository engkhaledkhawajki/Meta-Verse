from Codes.Models.SimpleMLP import SimpleMLP
from tensorflow.keras.models import Model
import tensorflow as tf


class Server:
    def __init__(self, input_shape, data_name, number_of_classes):
        self.data_name = data_name
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

    def build_server_model_for_mnist(self):
        """
         Builds MNIST model for use in MLP server. This is a helper function to be used in server_model. py
         
         :returns: server model for use in
        """
        smlp_global = SimpleMLP()
        global_model_favg = smlp_global.build(self.input_shape, self.number_of_classes)
        return global_model_favg

    def compile_fit_model(self, model, data, metrics, loss_function, percentage_to_train_on=0.1):
        """
         Compile and fit a model. This is the function that is called by fit_model. The data is passed as a list of 2 - tuples. The first tuple is the training data and the second tuple is the label, could be of type Tesnoreflow dataset.
         
         :param model: The model to be trained.
         :param data: The data to be used for training the model.
         :param metrics: The metrics to be used for training the model.
         :param loss_function: The loss function to be used for training the model.
         :param percentage_to_train_on: The percentage of data to be used for training.
         :returns: A tuple of ( loss_function data ) where loss_function is a function that takes a data and returns a loss
        """
        model.compile(loss=loss_function,
                      optimizer='Adam',
                      metrics=metrics)
        his = model.fit(data, epochs=5, verbose=1)

        return model

    def get_server_model_weights(self, model):
        """
         Returns the weights associated with the server model. This is a convenience method for calling the get_weights method of the given model and returning it.
         
         :param model: The model to get the weights for. Must be a : py : class : ` ~pysimm. amalgamation. Server `
         :returns: A list of : py : class : ` ~pysimm. amalgamation. Server `
        """
        return model.get_weights()

    def update_server_model_weights(self, model, weights):
        """
         Update weights of a server model. This is used to update the weights of a server model that is passed as an argument to the server_model method.
         
         :param model: The model to update. Must be of type ServerModel.
         :param weights: The new weights to set. Must be of type float.
         :returns: The updated model. Note that it is a copy of the model passed as an argument and will be returned
        """
        model = model.set_weights(weights)
        return model

    def build_model(self):
        """
         Build model for MNIST. This is a wrapper around build_server_model_for_mnist to be able to add a model to the server
         
         :returns: a dict with keys : data_name : name of the
        """
        if self.data_name == 'DAISEE':
            pass
        elif self.data_name == 'MNIST':
            return self.build_server_model_for_mnist()
