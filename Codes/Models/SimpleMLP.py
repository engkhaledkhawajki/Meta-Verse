from keras.layers import Dense, Activation
from keras.models import Sequential
from typing import Tuple


class SimpleMLP:
    @staticmethod
    def build(input_shape, num_classes) -> Sequential:
        """
         Build a Sequential model. It is used to train the neural network. The input_shape is the shape of the input to the model.
         
         :param input_shape: The shape of the input to the model.
         :param num_classes: The number of classes in the model.
         :returns: A Sequential model with the given parameters. Note that the model will be trained in - place and not in -
        """
        model = Sequential()
        model.add(Dense(200, input_shape=(input_shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))
        return model
