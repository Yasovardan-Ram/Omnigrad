# more imports..
from engine import Value
import random


''' This file creates a MLP from neurons and structuring 
them in layer which are then made structures as  MLP'''

class Neuron:
    #This class defines neuron wx + b  wher eeach set of input is gonna enter into them 
    
    def __init__(self,nin):
        # randomly initialize weights and bias 
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)] 
        self.bias =Value(random.uniform(-1,1))

    def __call__(self,x,act_type='Tanh'):
        

        if len(x) != len(self.w): # handles exception incase of reusing model and user inputs more or less no of inputs
            raise ValueError('incorrect no of inputs and weights')
        
        act_f= sum((wi*xi for wi,xi in zip(self.w,x)),self.bias)
    # to select the activation type
        if act_type == "Tanh":
            out=act_f.tanh()

        elif act_type == "RELU":
            out=act_f.RELU()
        
        elif act_type == "Leaky RELU":
            out=act_f.Leaky_RELU()
        
        elif act_type == "Sigmoid":
            out=act_f.sigmoid()
        
        elif act_type == "Swish":
            out=act_f.swish()
        
        elif act_type == "GELU":
            out=act_f.GELU()

        return out
    
    def parameters(self):# returns all the weight and bias in the neuron/layer/MLP
        return self.w + [self.bias]
    
# creates a layer to manage bunch of neurons 
# distributes the work to neurons
class Layer:

    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for i in range(nout)]
    
    def __call__(self,x,act_type):
        out=[n(x,act_type=act_type) for n in self.neurons]
        return out[0] if len(out)==1 else out
    
    def parameters(self): # returns all the weight and bias in the neuron/layer/MLP
        para = []
        for neuron in self.neurons:
            p=neuron.parameters()
            para.extend(p)
        return para


# creates a Multi Layer Perceptron to process the data 
class MLP:

    def __init__(self,nin,nouts): # here we get multiple outs as there might mulitple rows of inputs
        size=[nin]+nouts

        self.layers = [Layer(size[i],size[i+1]) for i in range(len(nouts))]

    
    def __call__(self,x,act_type):
        for layer in self.layers:
            x=layer(x,act_type=act_type)
        return x
    
    def parameters(self):# returns all the weight and bias in the neuron/layer/MLP
        return [p for layer in self.layers for p in layer.parameters()]


