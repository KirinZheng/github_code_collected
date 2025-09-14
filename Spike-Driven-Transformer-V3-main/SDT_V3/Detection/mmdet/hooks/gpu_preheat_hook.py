# mmdet/hooks/lock_memory_hook.py
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS  # 必须手动注册


@HOOKS.register_module()
class PreheatGPUHook(Hook):
    def __init__(self, gb=20):
        self.gb = gb

    def before_train(self, runner):
        num_bytes = int(self.gb * 1024 ** 3)
        num_elements = num_bytes // 4  # float32
        print(f'[PreheatGPUHook] Preallocating {self.gb:.1f} GB GPU memory temporarily...')
        buf = torch.empty(num_elements, dtype=torch.float32, device='cuda')
        del buf  # 🧠 删除后 PyTorch CUDA allocator 不会释放回系统，而是保留在内存池中
        torch.cuda.empty_cache()
        print(f'[PreheatGPUHook] Released buffer; CUDA memory pool warmed up.')
