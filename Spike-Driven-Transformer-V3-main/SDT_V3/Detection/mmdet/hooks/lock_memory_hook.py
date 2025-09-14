import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from torch.cuda.amp import autocast

@HOOKS.register_module()
class LockMemoryHook(Hook):
    def __init__(self, gb=2.0, dtype='auto'):  # 'auto', 'float32', or 'float16'
        self.gb = gb
        self.buffer = None
        self.dtype = dtype

    def before_train(self, runner):
        # 自动推测 dtype（推荐）
        if self.dtype == 'auto':
            use_fp16 = getattr(runner, 'fp16', False) or runner.cfg.get('fp16') is not None
            dtype = torch.float16 if use_fp16 else torch.float32
        elif self.dtype == 'float16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        num_bytes = int(self.gb * 1024 ** 3)
        num_elements = num_bytes // torch.tensor([], dtype=dtype).element_size()
        print(f"[LockMemoryHook] Locking {self.gb}GB as {dtype} tensor...")
        self.buffer = torch.empty(num_elements, dtype=dtype, device='cuda')
