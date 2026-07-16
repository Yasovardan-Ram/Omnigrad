# just some import statements...
import math
from graphviz import Digraph 
from PIL import Image
import io

Image.MAX_IMAGE_PIXELS = None # disables the limit of image size

class Value:
    # Custom Datatype is created that can handle all basic
    #  mathematical operation and function and backpropogate through it

    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self.prev=set(_children) # this will have the the set os parents which cause the the out 
        self.op =_op # reference its operation done like(add,sub,mul,etc)
        self._backward= lambda : None
        self.grad=0
        self.label = label
        

    def __repr__(self): # just return the data when printed
        return f'{self.data}'
    
    def __add__(self,other): # adds two values
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data + other.data,(self,other),'+')
    
        def _backward(): #finds gradient of add

            """we are doing += so we accumalate the gradient if we didnt and 
            incase of same value multiplied or added  the gradient is just going to reinitialize with same value since
            other and self will be same """
            self.grad += out.grad  
            other.grad += out.grad
        out._backward=_backward

        return out

    
    def __radd__(self,other): # add values incase they are in reverse 
        return self + other
    
    def __mul__(self,other):# multiplies values
        other=other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        """we are doing += so we accumalate the gradient if we didnt and 
            incase of same value multiplied or added  the gradient is just going to reinitialize with same value since
            other and self will be same """

        def _backward():# finds grad for mul
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self,other):# multiplies incase of reverse
        return self * other
    
    def __neg__(self): # unary (-)
        return self * -1
    
    def __sub__ (self,other):# subtract is simply negation of add
        return self + -other
    
    def __rsub__ (self,other):
        return other + -self # reverses the values to sub
    
    def __truediv__(self, other): #divides returns float we use mul to define it x/y=x * y^-1
        out = self * other **-1
        out.op='/'
        return out
    
    def __rtruediv__(self, other):# reverses the valeus for division
        out=other * self**-1
        out.op='/'
        return out
    
    def __pow__(self, other): # find power of number 
        other=other if isinstance(other,Value) else Value(other)
        out= Value(self.data**other.data,(self,other),'**')

# += similar explanation to add and mul
        def _backward(): #find gradient of power
            self.grad += (other.data*self.data**(other.data-1))*out.grad
        out._backward=_backward
        return out
    
    def exp(self): #finds power of number with respect e
        out = Value(math.exp(self.data), (self,), 'exp')
# += similar explanation to add and mul
        def _backward(): # find grad of exp which is just e 
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    #===============================
    #===== ACIVATION FUNCTION ======
    #===============================

    #funtions are defined using their formula and then their gradients
    
    def tanh(self): 
        x=self.data
        exp= (math.exp(2*x) -1)/(math.exp(2*x)+1)
        out =Value(exp,(self,),'tanh')


        def _backward():
            self.grad += (1-exp**2)*out.grad
        out._backward=_backward

        return out
    
    def RELU(self):  
        out= Value(max(0,self.data),_children=(self,),_op='RELU')

        def _backward():
            self.grad += (0 if self.data<=0  else 1.0)*out.grad
        out._backward =_backward
        return out
    
    def Leaky_RELU(self,alpha=0.01):
        out = Value(max(alpha*self.data,self.data)
                    ,_children=(self,),_op='L-RELU')
        
        def _backward():
            self.grad += (1.0 if self.data >0 else alpha)*out.grad
        out._backward=_backward
        return out
    
    def sigmoid(self):
        out= Value(1.0/(1.0+(-self).exp().data),_children=(self,),_op='Sigmoid')

        def _backward():
            self.grad += (out.data * (1 - out.data))*out.grad
        out._backward=_backward

        return out
    
    def swish(self):
        sig_data=1.0/(1.0+(-self).exp().data)
        out= Value(sig_data*self.data,
                   _children=(self,),_op='Swish')
        
        def _backward():
            self.grad += sig_data + out.data * (1 - sig_data) * out.grad
        out._backward=_backward

        return out
    
    def GELU(self):
        x = self.data
        cube=0.044715 * (x**3)
        inner= 0.79788456 * (x + cube)
        tanh_val=math.tanh(inner)
        out_data=0.5 * x * (1.0 + tanh_val)
        out= Value(out_data,_children=(self,),_op='GELU')

        def _backward():
            sech2 = 1.0 - (tanh_val ** 2)
            u_grad=0.79788456 * (1+0.134145 *(x **2))
            local_grad=0.5*((1+tanh_val) + x *(sech2 * u_grad))
            self.grad += local_grad * out.grad
        out._backward=_backward

        return out
    
    def abs_val(self): # give absolute of aue of a number stripping their sign
        return self.RELU() + (-self).RELU()

    def backward(self):

        # using depth first search to get all the nodes and calling _backward on it 
        topo=[]
        seen=set() # we use set here to efficiency reasons

        def topo_sort_dfs(x):
            if x not in seen:
                seen.add(x)
                for i in x.prev:
                    topo_sort_dfs(i)
                topo.append(x)

        topo_sort_dfs(self)        

        self.grad=1
        for node in reversed(topo):
            node._backward()

class visual:
    # This class is to draw the backprop flowchart using graphviz 

    def track(self,root):
        # this function is to get all nodes using depth first search algorithm 
        nodes,edges=set(),set()
        def dfs(x):
            if x not in nodes:
                nodes.add(x)

                for child in x.prev:
                    edges.add((child,x))
                    dfs(child)
        dfs(root)

        return nodes,edges

    def draw_dot(self,root):

        # Using graphviz to draw the flowchart using nodes form previous function

        self.dot = Digraph(graph_attr={'rankdir':'LR'})
        self.dot.attr(bgcolor='#E5E7EB')

        nodes,edges= self.track(root)

        # creates nodes based on track() connects the nodes to op node and then to out node
        for node in nodes:
            self.dot.node(name=str(id(node)),label= f'{{ {node.label} | Data: {node.data:.4f} | grad: {node.grad:.4f} }}',shape='Mrecord')

            if node.op: # depending their op they are we are connecting parent node to their custom op nodes
                op_id=str(id(node)) + node.op
                self.dot.node(name=op_id,label=node.op)

                self.dot.edge(op_id,str(id(node)))


        for ed1,ed2 in edges: # here wea re connecting the op node to the child ie the out node
            if ed2.op:
                op_id=str(id(ed2))+ed2.op
                self.dot.edge(str(id(ed1)),op_id)
                
        # getting the flowchart as png bytes and using io to virtually wrap it as file 
        self.png_bytes = self.dot.pipe(format='png')
        self.pil_image=Image.open(io.BytesIO(self.png_bytes))
        self.actual_width, self.actual_height = self.pil_image.size






