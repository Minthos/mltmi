import torch

class TanLU(torch.autograd.Function):
    """Hyperbolic tangens/linear unit. f(x) = x if x > 0, otherwise tanh(x).

    Has similar properties to ELU with alpha = 1 but seems cheaper to compute.
    Linear for x>0 and exponential otherwise. 0-centered."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.max(x, x.tanh())

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        derivative = (1.0 - x.tanh().pow(2))
        derivative[x >= 0] = 1
        return derivative * grad_output


class SinLU(torch.autograd.Function):
    """Sine/linear unit. f(x) = x if x > 0, otherwise sin(x).

    Computationally cheap. Linear and nonlinear. 0-centered.
    I don't know if it's better at fitting data than ELU/TanLU but maybe for some types of data."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.max(x, x.sin())

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return torch.max(x.abs() / x, x.cos()) * grad_output
        
