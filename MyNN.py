from MyTorch import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    

class Neuron(Module):
    def __init__(self, n_inputs, is_relu = True):
        self.n_inputs = n_inputs
        self.w = [Value(random.uniform(-1,1), name = "w") for _ in range(n_inputs)]
        self.b = Value(0, name = "b")
        self.is_relu = is_relu

    def __call__(self, x):
        res = self.b
        for i in range(self.n_inputs):
            res += self.w[i] * x[i]
        if self.is_relu:
            res = res.relu()
        return res
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.is_relu else 'Linear'} Neuron({len(self.w)})"
    
class Layer(Module):
    def __init__(self, n_inputs, n_outputs, is_relu = True):
        self.neurons = [Neuron(n_inputs=n_inputs,is_relu=is_relu) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out    
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, n_inputs, n_outputs):
        self.layers = [Layer(n_inputs=n_inputs, n_outputs=n_outputs[0], is_relu=True)]
        for i in range(0,len(n_outputs)-1):
            self.layers.append(Layer(n_inputs=n_outputs[i], n_outputs=n_outputs[i+1], is_relu=True))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

random.seed(1234)
lr = 0.00001
m = MLP(n_inputs=2, n_outputs=[40, 20, 1])

for i in range(100):
    sum_error = 0
    n_batch = 1000
    for j in range(n_batch):
        m.zero_grad()
        x = random.uniform(0, 5)
        y = random.uniform(0, 5)
        res = m([x, y])
        error = (Value(x*y) - res) * (Value(x*y) - res)
        error.backward()
        for p in m.parameters():
            p.data -= lr * p.grad
        sum_error += error.data
    print(sum_error/n_batch)