"""
Implementation of auto differentiation forward mode using Python

Based on MIT Open Courseware: 
https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/

In forward mode, the derivative is calculated along with forward pass of the function
"""
import numpy as np
import math

class DualNumber:
    """
    Dual number class which contains 2 attributes:
    1. value = value of the number
    2. derivative = derivative of the function evaluated at the given number

    the reverse operations (ex. __radd__) are spared as it adds little insight to the implementation
    """
    def __init__(self, value, grad = 1):
        self.value = value
        self.grad = grad

    def __add__(self, other):
        """
        Apply additional rule to derivative
        """
        new_value = self.value + other.value
        new_grad = self.grad + other.grad
        return DualNumber(new_value, new_grad)

    def __mul__(self, other):
        """
        Apply multiplication rule to derivative
        """
        new_value = self.value * other.value
        new_grad = self.value * other.grad + self.grad * other.grad
        return DualNumber(new_value, new_grad)

    def __div__(self, other):
        """
        Apply division rule to derivative
        """
        new_value = self.value / other.value
        new_grad = (other.value * self.grad + self.value * other.grad)/other.value ** 2
        return DualNumber(new_value, new_grad)

    def __sub__(self, other):
        """
        Apply subtraction rule to derivative
        """
        new_value = self.value - other.value
        new_grad = self.grad - other.grad
        return DualNumber(new_value, new_grad)
    
    def __str__(self):
        return str({
            "value": self.value,
            "derivative": self.grad
        })


# define mathematical functions for Dual Number classes - ensure derivatives are also captured inside

def sqrt(x: DualNumber, iter: int = 10) -> DualNumber:
    """
    Numerical approximation of the square root function using Heron's formula
    """
    t = 0.5*(1+x.value)
    t_prime = 1/2
    for i in range(1, iter):
        t = (t + x.value/t)/2
        t_prime = (t_prime + (t - x.value * t_prime)/t**2)/2
        i += 1
    return DualNumber(t, t_prime)


def power(x: DualNumber, order: int = 1) -> DualNumber:
    """
    y=x^order function
    """
    value = x.value**order
    derivative = order*(x.value**(order-1))
    return DualNumber(value, derivative * x.grad) #for chain rule


def sin(x: DualNumber) -> DualNumber:
    """
    Sinusoidal function
    """
    value = np.sin(x.value)
    derivative = np.cos(x.value)
    return DualNumber(value, derivative * x.grad) #for chain rule


def exp(x: DualNumber) -> DualNumber:
    """
    Exponential function
    """
    value = math.exp(x.value)
    derivative = math.exp(x.value)
    return DualNumber(value, derivative * x.grad) #for chain rule


if __name__=="__main__":
    print("Test implement f(x) = x^3 + sin (x^2)")
    print("f'(x) = 3x^2 + 2x*cos(x^2)")

    # Auto-differention version
    x = DualNumber(3)
    y = power(x, 3) + sin(power(x, 2))
    print(y)

    x = 3
    y = x**3 + np.sin(x**2)
    y_prime = 3*(x**2) + (2*x)*np.cos(x**2)
    print(f"value: {y}")
    print(f"derivative: {y_prime}")
