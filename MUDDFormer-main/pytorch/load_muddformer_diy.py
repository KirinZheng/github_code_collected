from muddformer.modeling_muddformer import MUDDFormer
from muddformer.configuration_muddformer import MUDDFormerConfig
import torch

## 加载网络模型
model = MUDDFormer(config=MUDDFormerConfig()).to('cuda')

## test for input
model.eval()
with torch.no_grad():
    B, S = 2, 16
    # 随机造点 token id（真实使用时请用你自己的 tokenizer）
    input_ids = torch.randint(0, 50432, (B, S), dtype=torch.long, device='cuda')
    with torch.device(torch.device('cuda:0')):
        model.setup_caches(max_batch_size=2, max_seq_length=2048, dtype=torch.float32)

    out = model(input_ids)      # 典型输出要么是 logits [B, S, vocab]，要么是 hidden [B, S, dim]
