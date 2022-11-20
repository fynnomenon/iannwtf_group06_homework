"""
@author: faurand, chardes, ehagensieker
"""
class MLP:
    """
    Multilayer Perceptron
    """

    def __init__(self):
        """
        Initializing the multilayer perceptron
        """
        self.layers = []

    def add_layer(self,layer):
        """
        Adding layers to the existing network
        """
        self.layers.append(layer)
  
    def forward_step(self,input_vars):
        """
        Input is forwarded through the network

        Args:
            i(array): input array

        Returns: 
            activation(array): activation of the current layer
        """
        activation = input_vars
        for layer in self.layers:
            activation = layer.forward_step(activation)

        return activation

    def backpropagation(self,loss,learning_rate):
        """
        Perform backpropagation by going bottom up through the network and 
        update the weights and biases.

        Args:
            loss(float): difference between target and prediction
            learning rate(float): value that determines to what degree we update the parameter
        """
        error_signal = loss
        for layer in reversed(self.layers):
            error_signal = layer.backward_step(error_signal,learning_rate)
    




    




