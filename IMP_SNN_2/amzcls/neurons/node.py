import math
from typing import Callable

import torch
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import (
    IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode,
    KLIFNode, PSN, MaskedPSN, SlidingPSN, GatedLIFNode
)
from torch import nn
from torch.nn import UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin

from .builder import NODES
from .surrogate import ZIF

for node in (IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode,
             KLIFNode, PSN, MaskedPSN, SlidingPSN, GatedLIFNode):
    NODES.register_module(module=node)


@NODES.register_module()
class TS1Node(nn.Module):
    def __init__(self, surrogate_function, threshold=1.):
        super(TS1Node, self).__init__()
        self.surrogate_function = surrogate_function
        self.threshold = threshold

    def forward(self, x):
        return self.surrogate_function(x - self.threshold)


@NODES.register_module()
class StateIFNode(IFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = StateIFNode(inputs.shape)
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, init_state_shape=(1, 1, 1, 1), v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False, sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(StateIFNode, self).__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.init_state = nn.Parameter(nn.init.uniform_(torch.empty(init_state_shape), a=0.4, b=0.6))
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x
        self.v += self.init_func(self.init_state)

    def reset(self):
        super(StateIFNode, self).reset()
        self.v += self.init_func(self.init_state)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.v = torch.broadcast_to(self.v, x.shape).to(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.v = torch.broadcast_to(self.v, x[0].shape).to(x)
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class StateLIFNode(LIFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = StateLIFNode(inputs.shape)
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, init_state_shape=(1, 1, 1, 1), tau: float = 2., decay_input: bool = True,
                 v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False,
                 sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(StateLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                           detach_reset, step_mode, backend, store_v_seq)
        # TODO: explore the best parameter
        self.init_state = nn.Parameter(nn.init.uniform_(torch.empty(init_state_shape), a=0.4, b=0.6))
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x
        self.v += self.init_func(self.init_state)

    def reset(self):
        super(StateLIFNode, self).reset()
        self.v += self.init_func(self.init_state)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.v = torch.broadcast_to(self.v, x.shape).to(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.v = torch.broadcast_to(self.v, x[0].shape).to(x)
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class LazyStateIFNode(IFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = LazyStateIFNode()
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False, sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(LazyStateIFNode, self).__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.init_state = None
        self.have_init = False
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x

    def check_init_state(self, x):
        if not self.have_init:
            self.init_state = nn.Parameter(nn.init.uniform_(
                torch.empty((1, *x.shape[1:]), device=x.device), a=0.4, b=0.6))
            self.have_init = True
        self.v = torch.broadcast_to(self.init_func(self.init_state), x.shape).to(x)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.check_init_state(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.check_init_state(x[0])
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class LazyStateLIFNode(LIFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = LazyStateLIFNode()
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False, sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(LazyStateLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                               detach_reset, step_mode, backend, store_v_seq)
        self.init_state = None
        self.have_init = False
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x

    def check_init_state(self, x):
        if not self.have_init:
            self.init_state = nn.Parameter(nn.init.uniform_(
                torch.empty((1, *x.shape[1:]), device=x.device), a=0.4, b=0.6))
            self.have_init = True
        self.v = torch.broadcast_to(self.init_func(self.init_state), x.shape).to(x)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.check_init_state(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.check_init_state(x[0])
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class LazyStatePSN(PSN):
    def __init__(
            self, time_step: int,
            surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(), sigmoid_init=False,
            **kwargs):
        super(LazyStatePSN, self).__init__(time_step, surrogate_function)
        self.init_state = None
        self.have_init = False
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x

        self.weight = nn.Parameter(
            torch.zeros([time_step, time_step + 1]))
        self.bias = nn.Parameter(
            torch.zeros([time_step, 1]))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)

    def check_init_state(self, x):
        if not self.have_init:
            # x_shape[T, B, ...]->init_shape[1, 1, ...]
            init_shape = (1, 1, *x.shape[2:])
            self.init_state = nn.Parameter(nn.init.uniform_(
                torch.empty(init_shape, device=x.device), a=-0.2, b=0.2))
            self.have_init = True
        init_state = torch.broadcast_to(
            self.init_func(self.init_state), (1, *x.shape[1:])
        )  # init_state[1, 1, ...]->init_state[1, B, ...]
        return init_state.to(x)

    def forward(self, x):
        init_state = self.check_init_state(x)  #
        # (init_state[1, B, ...], x[T, B, ...])-> h_seq[T+1, B, ...]
        h_seq = torch.concat([init_state, x], dim=0)
        # weight[T, T+1] @ h_seq[T+1, B, ...] -> h_seq[T, B, ...]
        h_seq = torch.addmm(self.bias, self.weight, h_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x.shape)


# Todo: MMEngine Not Support in Training
@NODES.register_module()
class LazyStateLIFNodeBeta(LazyModuleMixin, LIFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = LazyStateIFNode()
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False, device=None, dtype=None, sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(LazyStateLIFNodeBeta, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                                   detach_reset, step_mode, backend, store_v_seq)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.init_state = UninitializedParameter(**factory_kwargs)
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x

    def reset(self):
        super(LazyStateLIFNodeBeta, self).reset()
        self.v += self.init_func(self.init_state)

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        pass

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.v = torch.broadcast_to(self.v, x.shape).to(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.v = torch.broadcast_to(self.v, x.shape[1:]).to(x)
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


# Todo: MMEngine Not Support in Training
@NODES.register_module()
class LazyStateIFNodeBeta(LazyModuleMixin, IFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = LazyStateIFNode()
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False, device=None, dtype=None, sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(LazyStateIFNodeBeta, self).__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.init_state = UninitializedParameter(**factory_kwargs)
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x

    def reset(self):
        super(LazyModuleMixin, self).reset()
        self.v += self.init_func(self.init_state)

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        pass

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.v = torch.broadcast_to(self.v, x.shape).to(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.reset()
            self.v = torch.broadcast_to(self.v, x.shape[1:]).to(x)
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class LazyStateChannelLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True,
                 v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        super(LazyStateChannelLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                                      detach_reset, step_mode, backend, store_v_seq)
        self.have_init = False
        self.init_state = None

    def check_init_state(self, x):
        if not self.have_init:
            self.init_state = nn.Parameter(nn.init.uniform_(
                torch.empty((1, x.shape[1], 1, 1), device=x.device), a=-0.2, b=0.2))
            self.have_init = True
        self.v = torch.broadcast_to(self.init_func(self.init_state), x.shape).to(x)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.check_init_state(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.check_init_state(x[0])
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class LazyStateHWLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True,
                 v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        super(LazyStateHWLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                                 detach_reset, step_mode, backend, store_v_seq)
        self.have_init = False
        self.init_state = None

    def check_init_state(self, x):
        if not self.have_init:
            self.init_state = nn.Parameter(nn.init.uniform_(
                torch.empty((1, 1, *x.shape[2:]), device=x.device), a=-0.2, b=0.2))
            self.have_init = True
        self.v = torch.broadcast_to(self.init_func(self.init_state), x.shape).to(x)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.check_init_state(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.check_init_state(x[0])
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class RandNStateLIFNode(LIFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = LazyStateLIFNode()
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, tau: float = 2., decay_input: bool = True,
                 v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        super(RandNStateLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                                detach_reset, step_mode, backend, store_v_seq)
        self.mu = nn.Parameter(nn.init.uniform_(torch.tensor(0.), -0.1, +0.1))
        self.sigma = nn.Parameter(torch.tensor(1.))

    def random_state(self, x):
        # x[B, C, H, W]->random_state[B, C, H, W]
        rand = torch.randn(x.shape, device=x.device).to(x)
        self.v += torch.clip(rand * self.sigma + self.mu, min=-1, max=1)

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.random_state(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.random_state(x[0])
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class LIFLegacy(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFLegacy, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma

    def forward(self, x):
        mem_v = []
        mem = 0
        for t in range(x.shape[0]):
            mem = self.tau * mem + x[t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)

        return torch.stack(mem_v)


@NODES.register_module()
class StateLIFLegacy(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(StateLIFLegacy, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem = None
        self.have_init = False

    def init_membrane_state(self, x):
        if not self.have_init:
            # shape[T, B, C, H, W]->init_shape[1, C, H, W]
            init_shape = (1, *x.shape[2:])
            self.mem = nn.Parameter(nn.init.uniform_(
                torch.empty(init_shape, device=x.device), a=-0.2, b=0.2))
            self.have_init = True
            print('==================init==================')
        return self.mem.to(x)

    def forward(self, x):
        mem = self.init_membrane_state(x)
        mem_v = []
        for t in range(x.shape[0]):
            mem = self.tau * mem + x[t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)

        return torch.stack(mem_v)
