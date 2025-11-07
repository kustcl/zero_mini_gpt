import torch
from gpt_model import GPT2
import tiktoken

class LLMConfig:
    d_model: int = 768
    context_length: int = 100
    num_heads: int = 12
    head_size: int = d_model // num_heads
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout: float = 0.1
    vocab_size: int = 50257
    n_layer: int = 6
gpt_config=LLMConfig()
device='cuda' if torch.cuda.is_available() else 'cpu'
model=GPT2(config=gpt_config)
model.load_state_dict(torch.load('gpt2.pt',map_location=device))
model.to(device=device)
model.eval()
tokenizer=tiktoken.get_encoding('gpt2')
input='You are glad'
input_ids=tokenizer.encode(input)
input_tensor=torch.tensor(input_ids,device=device)
input_tensor=input_tensor.unsqueeze(dim=0)
with torch.no_grad():
    output_ids=model.generate(input_tensor,max_new_tokens=100,temperature=0.6)
generate_text=tokenizer.decode(output_ids[0].tolist())
print(f'----------input text：{input}------------')
print(f'----------mini_gpt_generate_text: {generate_text}-------------------')


