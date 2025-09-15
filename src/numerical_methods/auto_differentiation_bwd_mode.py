"""
Implementation of backwards auto-differentiation

Other names include:
- backpropagation
- backward mode

Based on MIT Open Courseware: 
https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/

Unlike forward auto-differentiation, 2 passes are needed to calculate the derivative:
1. Calculating value (while keeping track what operations/functions it went through)
2. Backtracing to calculate the derivative along the way
"""

import math
from typing import Callable

class TrackedDualNumber:
    """
    Tracking of operations performed is required because the gradient will be calculated with the 2nd pass using backward function

    Similar to forward model implementation reverse methods (ex. __radd__) are excluded as this is meant to be for understanding
    """
    def __init__(self, value: float, grad_fn: Callable | None = None):
        self.value = float(value)
        self.grad = 0.0
        self.grad_fn = grad_fn
        self.parents = [] # upstream operation dependencies of this object

    def backward(self):
        """
        backpropagation calculation
        """
        calculation_history = [] #this should be accumulated in the order of all operations of a function

        def build_calculation_history(num):
            """
            helper function to build a topological ordered list of nodes of the computations conducted up to and inclusive of itself
            """
            if num in calculation_history:
                return
            for parent in num.parents:
                build_calculation_history(parent)
            calculation_history.append(num)

        build_calculation_history(self)

        for node in calculation_history: # zero out all histories
            node.grad = 0.0
        calculation_history[-1].grad = 1.0 # initialize the first step of propagation to be 1

        for node in reversed(calculation_history):
           #note that variable may appear multiple times in which the memory the previous gradient will be used to add on to the additional gradient calculations automatic with nature of python
           if node.grad_fn is not None:
               node.grad_fn(node.grad) 

    def __repr__(self):
        return str({"value": self.value, "grad": self.grad})

    def __add__(self, other):
        new_value = self.value + other.value
        result = TrackedDualNumber(new_value)
        result.parents = [self, other] # stores the history

        def grad_fn(input):
            self.grad += input #with respect to self
            other.grad += input #with respect to other
            print(f"self: {self.grad}")
            print(f"other: {other.grad}")
        
        result.grad_fn = grad_fn #everytime an operation is performed, instead of calculating and saving the gradient, the gradient function itself is saved
        return result
    
    def __sub__(self, other):
        result = TrackedDualNumber(self.value - other.value)
        result.parents = [self, other]
        
        def grad_fn(input):
            self.grad += input
            other.grad -= input
            print(f"self: {self.grad}")
            print(f"other: {other.grad}")
        
        result.grad_fn = grad_fn
        return result
    
    def __mul__(self, other):
        result = TrackedDualNumber(self.value * other.value)
        result.parents = [self, other]

        def grad_fn(input): #multiplication rule
            self.grad += input * other.value
            other.grad += input * self.value
            print(f"self: {self.grad}")
            print(f"other: {other.grad}")

        result.grad_fn = grad_fn
        return result
    
    def __div__(self, other):
        result = TrackedDualNumber(self.value / other.value)
        result.parents = [self, other]

        def grad_fn(input): #quotient rule
            self.grad += input / other.value
            other.grad += input * self.value / (other.value ** 2)
            print(f"self: {self.grad}")
            print(f"other: {other.grad}")

        result.grad_fn = grad_fn
        return result
    
    def __pow__(self, other):
        result = TrackedDualNumber(self.value * other.value)
        result.parents = [self, other]
        
        def grad_fn(input):
            self.grad += input * other.value * self.value ** (other.value - 1)
            other.grad = 0
            print(f"self: {self.grad}")

        result.grad_fn = grad_fn
        return result

def sin(a: TrackedDualNumber) -> TrackedDualNumber:
    result = TrackedDualNumber(math.sin(a.value))
    result.parents = [a]

    def grad_fn(input):
        a.grad += input * math.cos(a.value)
        print(f"self: {a.grad}")

    result.grad_fn = grad_fn
    return result
    
    
if __name__=="__main__":
    x = TrackedDualNumber(3.0)
    y = TrackedDualNumber(2.0)
    
    z = x * y + sin(x) - y ** TrackedDualNumber(2.0)
    print(f"z = x * y + sin(x) - y^2")
    print(f"x = {x.value}, y = {y.value}")
    print(f"z = {z.value:.4f}")
    
    z.backward()
    print(f"dz/dx = {x.grad:.4f}")
    print(f"dz/dy = {y.grad:.4f}")
    print(f"z.grad = {z.grad}")
    print()