class Value:

    def __init__(self, data, name = "", parents=(), operation = None):
        self.data = data
        self.name = name
        self.grad = 0
        self.parents = parents
        self.operation = operation
        self._backward = lambda: None

    def __repr__(self):
        return f"Name={self.name}, Value={self.data}, Grad={self.grad} Operation={self.operation}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data = self.data + other.data,
            parents = (self, other),
            operation = '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data = self.data * other.data,
            parents = (self, other),
            operation = '*')
        
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out        
    
    def backward(self):
        self.grad = 1
        topo = []

        def find_topo(node):
            if node in topo:
                pass
            else:
                for p in node.parents:
                    find_topo(p)
                topo.append(node)
        find_topo(self)
        topo.reverse()
        for node in topo:
            node._backward()

    def relu(self):
        if self.data < 0:
            val = -self.data
        else:
            val = self.data
        out = Value(data = val, parents=(self, ), operation="Relu")

        def _backward():
            if self.data > 0:
                self.grad += out.grad
        out._backward = _backward

        return out


    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1     