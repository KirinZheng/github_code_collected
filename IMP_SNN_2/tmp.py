from typing import Callable
# from spikingjelly.activation_based import surrogate
# from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.clock_driven.neuron import LIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import surrogate, lava_exchange
import logging
try:
    import cupy
    from spikingjelly.clock_driven import neuron_kernel, cu_kernel_opt
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cu_kernel_opt = None

from torch import nn
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn import UninitializedParameter

class MultiStepLazyStateLIFNode(MultiStepLIFNode):
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

    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6,
                 sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(MultiStepLazyStateLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                               detach_reset, backend, lava_s_cale)
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

        x_seq = args[0]  # select first arguments
        single_mode = args[1]

        if not single_mode:
            self.check_init_state(x_seq[0])  # 

        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super(MultiStepLIFNode, self).forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)

        # if self.step_mode == 's':
        #     self.check_init_state(x)
        #     return self.single_step_forward(*args, **kwargs)
        # elif self.step_mode == 'm':
        #     self.check_init_state(x[0])
        #     return self.multi_step_forward(*args, **kwargs)
        # else:
        #     raise ValueError(self.step_mode)



class MultiStepLazyStateLIFNodeBeta(LazyModuleMixin, MultiStepLIFNode):
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

    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6, 
                 device=None, dtype=None, sigmoid_init=False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(MultiStepLazyStateLIFNodeBeta, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                               detach_reset, backend, lava_s_cale)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.init_state = UninitializedParameter(**factory_kwargs)
        self.init_func = torch.sigmoid if sigmoid_init else lambda x: x

    def reset(self):
        super(MultiStepLazyStateLIFNodeBeta, self).reset()
        self.v += self.init_func(self.init_state)

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        pass

    def forward(self, *args, **kwargs):

        x_seq = args[0]  # select first arguments
        # single_mode = args[1]

        # if not single_mode:
        print("here1")
        self.v = torch.broadcast_to(self.v, x_seq.shape[1:]).to(x_seq)  # 

        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super(MultiStepLIFNode, self).forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)

        # if self.step_mode == 's':
        #     self.check_init_state(x)
        #     return self.single_step_forward(*args, **kwargs)
        # elif self.step_mode == 'm':
        #     self.check_init_state(x[0])
        #     return self.multi_step_forward(*args, **kwargs)
        # else:
        #     raise ValueError(self.step_mode)


if __name__ == "__main__":
    inputs = torch.randn(2, 1, 4)
    # node = MultiStepLazyStateLIFNode(tau=2.0, detach_reset=True, backend='torch')
    node = MultiStepLazyStateLIFNodeBeta(tau=2.0, detach_reset=True, backend='torch')
    print(node.v)

    out_1 = node(inputs, False)
    print(node.v)
    print(node.init_state)
    node.reset()
    print(node.v)
    print(node.init_state)
    # out_2 = []
    # for i in range(2):
    #     if i == 0:
    #         tmp_out = node(inputs[i].unsqueeze(0), True).squeeze(0)
    #     else:
    #         tmp_out = node(inputs[i].unsqueeze(0), False).squeeze(0)
    #     out_2.append(tmp_out)
    # out_2 = torch.stack(out_2, dim =1)
    # print(node.v)
    # print(node.init_state)
    # print(out_1 == out_2)


    # from spikingjelly.clock_driven.neuron import MultiStepLIFNode

    # tmp_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    # input_tt = torch.randn(3,2,4)

    # output_tt = tmp_lif(input_tt)