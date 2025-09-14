import torch
from spikingjelly.activation_based.surrogate import (
    PiecewiseQuadratic, PiecewiseExp, Sigmoid, SoftSign, ATan, NonzeroSignLogAbs,
    Erf, PiecewiseLeakyReLU, SquarewaveFourierSeries, S2NN, QPseudoSpike, LeakyKReLU,
    FakeNumericalGradient, LogTailedReLU
)

from .builder import SURROGATE

for surrogate in (
        PiecewiseQuadratic, PiecewiseExp, Sigmoid, SoftSign, ATan, NonzeroSignLogAbs,
        Erf, PiecewiseLeakyReLU, SquarewaveFourierSeries, S2NN, QPseudoSpike, LeakyKReLU,
        FakeNumericalGradient, LogTailedReLU
):
    SURROGATE.register_module(module=surrogate)


# class ZIF(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, gama):
#         out = (input > 0).float()
#         L = torch.tensor([gama])
#         ctx.save_for_backward(input, out, L)
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         (input, out, others) = ctx.saved_tensors
#         gama = others[0].item()
#         grad_input = grad_output.clone()
#         tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
#         grad_input = grad_input * tmp
#         return grad_input, None


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        factor = ctx.alpha - ctx.saved_tensors[0].abs()
        grad *= (1 / ctx.alpha) ** 2 * factor.clamp(min=0)
        return grad, None
