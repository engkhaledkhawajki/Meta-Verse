import math
import numpy as np


class DP:
    def __init__(self, number_of_clients: int, clipping_norm: float, epsilon: float, comms_round: int,
                 user_per_round: int, data_size: int, learning_rate: float, momentum: float) -> None:
        # Hyperparameters
        self.gamma = None
        self.number_of_clients = number_of_clients
        self.clipping_norm = clipping_norm
        self.epsilon = epsilon
        self.comms_round = comms_round
        self.user_per_round = user_per_round
        self.data_size = data_size
        self.delta = (1 / self.data_size) * 0.5
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.current_epoch_num = 0
        self.c = math.sqrt(2.0 * math.log(1.25 / self.delta))

    def clip_weights(self, weight: np.ndarray) -> np.ndarray:
        """
         Clip weights to max value. This is a helper for L { minimize } and L { minimize_cross }.
         
         :param weight: Weights to be clipped. The weights should be in the range [ 0 1 ].
         :returns: Clipped weight ( s ). Weights are clipped to the range [ 0 1 ] by dividing by clipping_norm
        """
        # Calculate the maximum value of the weight.
        for idx, w in enumerate(weight):
            weight[idx] = np.array(weight[idx], dtype=np.float32)
            max_value = np.maximum(1.0, weight / self.clipping_norm)
        weight /= max_value  # Element-wise division
        return weight

    def get_global_stddev(self, current_epoch_num: int) -> float:
        """
         Calculates the standard deviation of the client's data. It is based on equation 2 in Allen et al.
         
         :param current_epoch_num: number of the current epoch :returns: standard deviation of the client's data in
         the current epoch as a float between 0 and 1. This is used to calculate the mean
        """
        self.gamma = -1 * np.log(1 - (self.user_per_round / self.number_of_clients) + (
                self.user_per_round / self.number_of_clients) * np.exp(
            (-1 * self.epsilon) / math.sqrt(self.user_per_round)))
        if (current_epoch_num + 1) > (self.epsilon / self.gamma):
            b = -1 * ((current_epoch_num + 1) / self.epsilon) * np.log(
                1 - (self.number_of_clients / self.user_per_round) + (
                        (self.number_of_clients / self.user_per_round) * np.exp((-1 * self.epsilon) / (current_epoch_num + 1))))
            sqrt_arg = abs(
                (((current_epoch_num + 1) ** 2) / (b ** 2)) - self.user_per_round * ((current_epoch_num + 1) ** 2))

            # Now, you can safely take the square root
            stddev = (2 * self.c * self.clipping_norm * math.sqrt(sqrt_arg)) / (
                    (self.data_size / self.number_of_clients) * self.user_per_round * self.epsilon)

            # stddev = (2 * self.c * self.clipping_norm * math.sqrt((((current_epoch_num + 1) ** 2) / (b ** 2)) - self.user_per_round * ((current_epoch_num + 1)**2) )) / ((self.data_size / self.number_of_clients) * self.user_per_round * self.epsilon)
        else:
            stddev = 0
        return stddev

    def get_c_value(self) -> float:
        """
         Returns the c - value of the spline. It is defined as sqrt ( 2. 25 / delta )
         
         :returns: The c - value
        """
        return math.sqrt(2.0 * math.log(1.25 / self.delta))

    def get_epsilon(self) -> float:
        """
         Get the value of : attr : ` epsilon `. This is a constant that can be used to set the tolerance of the linear programming problem.
         
         :returns: the value of : attr : ` epsilon ` or None if not set by the user 
        """
        return self.epsilon

    def get_clipping_norm(self) -> float:
        """
         Get the norm of the clipping region. This is used to determine how much space to clip the image before resizing it to fit the image's size.
         
         :returns: The norm of the clipping region in units of : math : ` m^ { - 1 } `
        """
        return self.clipping_norm
