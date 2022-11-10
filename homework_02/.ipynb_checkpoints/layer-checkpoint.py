"""
@authors: faurand, chardes, ehagensieker
"""
import numpy as np

class Layer:
    """
    A single, fully-connected layer 
    """

    def __init__(self,n_units,input_units):
        """
        Initializing a single, fully-connected layer
        """
        self.bias_vector = np.zeros(n_units)
        self.weight_matrix = np.random.uniform(0, 1,(input_units,n_units))
        #self.weight_matrix = np.random.rand(input_units,n_units)
        self.layer_input = np.empty(input_units)
        self.layer_preactivation = np.empty(n_units)
        self.layer_activation = np.empty([n_units,1])

    def forward_step(self, layer_input):
        """
        The activation of the perceptron is calculated using Relu

        Args: 
          layer_output(array): inputs to the perceptron

        Returns: 
          self.layer_activation (array): Outputs of the perceptrons in the layer
        """
        self.layer_input = layer_input
        self.layer_preactivation =  self.layer_input @ self.weight_matrix + self.bias_vector

        self.layer_activation = np.where(self.layer_preactivation > 0, self.layer_preactivation, 0)

        return self.layer_activation

    def backward_step(self,error_signal,learning_rate):
        """
        Computation of the derivatives and gradients for the backward step, which is used
        for updating the parameters. 

        Args:
          error_signal(float): Error calculated with the loss function between target and prediction
          learning_rate(float): value that determines to what degree we update the parameter

        Returns: 
          error(float): value that determines the error 
        """
        gradient_preactivation = error_signal * np.where(self.layer_preactivation > 0, 1, 0)

        gradient_weights = self.layer_input.T @ gradient_preactivation
        gradient_bias = gradient_preactivation
        
        error = gradient_preactivation @ self.weight_matrix.T
        
        self.weight_matrix = self.weight_matrix - learning_rate*gradient_weights 
        self.bias_vector = self.bias_vector - learning_rate*gradient_bias

        return error

  
    


