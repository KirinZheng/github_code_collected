from mmengine.hooks import Hook
from spikingjelly.activation_based.functional import reset_net

from amzcls.registry import HOOKS


@HOOKS.register_module()
class ResetSpikeNeuron(Hook):

    def __init__(self, ):
        super().__init__()

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        reset_net(runner.model)
