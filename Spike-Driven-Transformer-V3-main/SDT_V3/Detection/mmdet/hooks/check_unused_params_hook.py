# my_project/hooks/check_unused_params_hook.py

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class CheckUnusedParamsHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        model = runner.model
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"[⚠️ WARNING] No gradient for: {name}")
