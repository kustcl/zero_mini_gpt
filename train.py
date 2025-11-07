import torch
from torch.utils.data import DataLoader, Dataset, random_split
import tiktoken
from dataclasses import dataclass
from gpt_model import GPT2
from torch import optim


@dataclass
class LLMConfig:
    d_model: int = 768
    context_length: int = 100
    num_heads: int = 12
    head_size: int = d_model // num_heads
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout: float = 0.1
    vocab_size: int = 50257
    n_layer: int = 6


@dataclass
class TrainConfig:
    batch_size: int = 16
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 3e-5
    epochs: int = 100


train_config = TrainConfig()

gpt_config = LLMConfig()

with open('', 'r',encoding='utf-8') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')


class MyDataset(Dataset):
    def __init__(self, text, tokenizer, all_size, config):
        self.idx = []
        self.targets = []
        token_ids = tokenizer.encode(text)
        start_token_index = torch.randint(0, high=len(token_ids) - config.context_length, size=(all_size,))
        for i in start_token_index:
            input_chunk = token_ids[i:i + config.context_length]
            target_chunk = token_ids[i + 1:i + config.context_length + 1]
            self.idx.append(input_chunk)
            self.targets.append(target_chunk)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        return torch.tensor(self.idx[item]), torch.tensor(self.targets[item])


all_dataset = MyDataset(text, tokenizer, all_size=1000,config=gpt_config)
train_dataset, val_dataset = random_split(all_dataset, lengths=[0.9, 0.1])
train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2 * train_config.batch_size, shuffle=False)

model = GPT2(config=gpt_config)
model.to(train_config.device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total Params:{total_params / 1e6}M')

optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)


def train(model, optimizer, scheduler, train_loader, train_config):
    model.train()
    total_loss = 0
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(train_config.device), y.to(train_config.device)
        _, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss


def val(model, val_loader, train_config):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x, y = x.to(train_config.device), y.to(train_config.device)
            _, loss = model(x, targets=y)
            total_loss += loss.item()
        return total_loss


for epoch in range(train_config.epochs):
    train_loss = train(model, optimizer, scheduler, train_loader, train_config)
    val_loss = val(model, val_loader, train_config)
    print(f'epoch {epoch},train_loss {train_loss / len(train_loader): .4f} val_loss {val_loss / len(val_loader): .4f}')

torch.save(model.state_dict(), 'gpt2.pt')
