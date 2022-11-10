"""
@authors: faurand, chardes, ehagensieker
"""
import numpy as np
class Layer:

  """
  A single Layer 
  """

  def __init__(self,n_units,input_units):
    self.bias_vector = np.zeros(n_units)
    self.weight_matrix = np.random.rand(input_units,n_units)
    self.layer_input = np.empty(input_units)
    self.layer_preactivation = np.empty(n_units)
    self.layer_activation = np.empty([n_units,1])

   

  def forward_step(self, a):

    """
    The activation of the perceptron is calculated using Relu

    Args: 
      a(array): inputs to the perceptron

    Returns: 
      self.layer_activation (array): Outputs of the perceptrons in the layer
    """

    self.layer_input = a
  
    self.layer_preactivation = np.sum(self.weight_matrix * self.layer_input + self.bias_vector, axis=0)
  
    for i,x in enumerate(self.layer_preactivation):
      value = max(0,x)
      self.layer_activation[i] = [value]
  
    return self.layer_activation


  def backward_step(self,error_signal, learning_rate):
    """
    Computation of the derivatives and gradients for the backward step, which is used
    for updating the parameters. 

    Args:
      error_signal(float): Error calculated with the loss function between target and prediction
      learning_rate(float): value that determines to what degree we update the parameter

    Returns: 
      error(float): value that determines the error 
    """

    derivative = lambda x: 1 if x > 0 else 0
    derived = np.array([derivative(x) for x in self.layer_preactivation])
    error_sig = error_signal * derived
    
    gradient_weights = self.layer_input * error_sig

    gradient_bias = error_sig
    
    error = np.sum(error_sig * self.weight_matrix, axis=1)
    
    self.weight_matrix = self.weight_matrix - learning_rate*gradient_weights 
    
    self.bias_vector = self.bias_vector - learning_rate*gradient_bias
    

    return error

  
    

