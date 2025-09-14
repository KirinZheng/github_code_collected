# MUDDFormer

This folder contains pytorch implementations of MUDDPythia and MUDDFormer. We release model checkpoints of [(MUDDFormer-2.8B)](https://huggingface.co/Caiyun-AI/MUDDFormer-2.8B), [(MUDDPythia-1.4B)](https://huggingface.co/Caiyun-AI/MUDDPythia-1.4B) and [(MUDDPythia-2.8B)](https://huggingface.co/Caiyun-AI/MUDDPythia-2.8B) on Huggingface🤗.
All of these models are pretrained on the Pile with 300B tokens. We propose MUltiway Dynamic Dense (MUDD) connections, a simple yet effective method to address the limitations of residual connections and enhance cross-layer information flow in Transformers. Please see downstrem evaluations and more details in the paper[(MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections)](https://arxiv.org/abs/2502.12170). For training MUDDFormer efficiently, we provide Jax code in the [(jax folder)](https://github.com/Caiyun-AI/MUDDFormer/tree/main/jax).

We recommend <strong>compiled version</strong> of MUDDFormer with *torch.compile* for inference acceleration. Please refer to Generation section for compile implementation.

## Usage

### Env

```
pip install transformers==4.40.2 torch==2.5.1 einops==0.8.0
```


### Generation 

```
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

device = torch.device('cuda:0')
dtype = torch.bfloat16
MAX_BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
NUM_TOKENS_TO_GENERATE = 10
COMPILE = True
OPTIMIZED_COMPILE = False 

if OPTIMIZED_COMPILE: 
    import torch._dynamo.config
    import torch._inductor.config
    torch._dynamo.config.cache_size_limit = 64
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True

tokenizer = AutoTokenizer.from_pretrained("Caiyun-AI/MUDDFormer-2.8B")
model = AutoModelForCausalLM.from_pretrained("Caiyun-AI/MUDDFormer-2.8B", trust_remote_code=True)

_ = model.to(device=device,dtype=dtype)
with torch.device(device):
    model.setup_caches(max_batch_size=MAX_BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH, dtype=dtype)

def decode_one_token(model, cur_token, input_pos):
    logits = model(cur_token, input_pos=input_pos, return_tensor=True)
    new_token = torch.argmax(logits[:, -1], dim=-1)[:,None]
    return new_token

prompt = "Beijing is the capital of China. London is the capital of"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

compiled_decode_one_token = torch.compile(decode_one_token,mode="reduce-overhead", fullgraph=True) if COMPILE else None

print('Start generating tokens, but it will take a few minutes to compile at the first time.')
for i in range(10):
    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(input_ids.to(device),num_tokens_to_generate=NUM_TOKENS_TO_GENERATE, compiled_decode_one_token=compiled_decode_one_token)
        text = tokenizer.decode(generated_ids[0])
        if i ==0:
            print(f'Generated text: {text}')
    t1 = time.time()
    print(f'Time consumed at iteration {i}: {t1-t0}s')
```
