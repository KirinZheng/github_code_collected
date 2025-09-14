from loralib.utils import mark_only_lora_as_trainable
from mmengine import MMLogger
from mmengine.hooks import Hook

from amzcls.registry import HOOKS


@HOOKS.register_module()
class LoRAHook(Hook):
    def __init__(self, fine_tuning_key: list = None, bias: str = 'none'):
        """
        :param fine_tuning_key: ['backbone', 'neck', 'head']
        :param bias: ['none', 'all', 'lora_only']
        """
        super().__init__()
        self.fine_tuning_key = ['backbone'] if fine_tuning_key is None else fine_tuning_key
        assert bias in ['none', 'all', 'lora_only'], MMLogger.get_current_instance().info(
            '[AMZCLS] [LoRAHook] bias should be in [none, all, lora_only]')

    def before_train(self, runner) -> None:
        MMLogger.get_current_instance().info(f'[AMZCLS] [LoRAHook] {self.fine_tuning_key}.')
        for key in self.fine_tuning_key:
            if hasattr(runner.model, key):
                mark_only_lora_as_trainable(getattr(runner.model, key))


@HOOKS.register_module()
class RankAscent(Hook):

    def __init__(self, fine_tuning_key: list = None):
        """
        :param fine_tuning_key: ['backbone', 'neck', 'head']
        """
        super().__init__()
        self.fine_tuning_key = ['backbone'] if fine_tuning_key is None else fine_tuning_key

    def before_train(self, runner) -> None:
        MMLogger.get_current_instance().info(f'[AMZCLS] [LoRAHook] {self.fine_tuning_key}.')
