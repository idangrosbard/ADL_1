from lstm.lstm_lm import LSTM_LM
import torch
from datasets import load_dataset
import torch.utils
from torch.utils import data
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def setup_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("Kristijan/wikitext-103-tokenizer")
    tokenizer.add_special_tokens({"mask_token": "<MASK>"})
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    
    return tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = [s for s in dataset if len(s['text']) > 0]
        self.tokenizer = setup_tokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # add CLS
        samp = self.dataset[idx]['text']
        return self.tokenizer(samp)
        # return [x.long() for x in self.tokenizer(self.dataset[idx]['text'])]


def setup_dataloaders(batch_size: int = 2):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    
    train_ds = Dataset(ds['train'])
    eval_ds = Dataset(ds['validation'])
    test_ds = Dataset(ds['test'])
    dc = DataCollatorForLanguageModeling(tokenizer=train_ds.tokenizer, mlm=True, mlm_probability=0.15)

    train_dl = data.DataLoader(train_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    eval_dl = data.DataLoader(eval_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(test_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)

    return train_dl, eval_dl, test_dl


def do_batch(model, batch, optimizer, loss_fn, writer: SummaryWriter, device, train: bool = True):
    optimizer.zero_grad()
    b_inp = batch['input_ids'].to(device, non_blocking=True).long()
    labels = batch['labels'].long().to(device, non_blocking=True)

    h, c = None, None
    for t in range(1, labels.shape[1]):
        logits, h, c = model(b_inp, h, c)
        loss = loss_fn(logits, labels[:, t])
    
    if train:
        loss.backward()
        optimizer.step()
    writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', loss.item())
    writer.flush()
    return loss.item()
    

def do_epoch(model, dataloader, optimizer, loss, writer: SummaryWriter, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    for batch in tqdm(dataloader, desc='train' if train else 'eval'):
        total_loss += do_batch(model, batch, optimizer, loss, writer, device, train) / len(dataloader)
    writer.add_scalar(f'{"train" if train else "eval"}/epoch_loss', total_loss)
    writer.flush()
    return total_loss


def train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, loss_fn, writer: SummaryWriter, device, epochs: int = 1, eval_every: int = 1):
    for e in range(epochs):
        total_loss = do_epoch(model, train_dataloader, optimizer, loss_fn, writer, device)
        print(f'train loss: {total_loss}')
        
        if e % eval_every == 0:
            total_loss = do_epoch(model, eval_dataloader, optimizer, loss_fn, writer, device, train=False)
            print(f'eval loss: {total_loss}')
    
    total_loss = do_epoch(model, test_dataloader, optimizer, loss_fn, writer, device, train=False)
    print(f'Test loss: {total_loss}')
    return model

if __name__ == '__main__':
    N_epochs = 10
    bsize = 64
    lr = 1e-5
    max_lr = 1e-4
    model_d = 16
    hidden_d = 64
    n_layers = 3
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl, eval_dl, test_dl = setup_dataloaders(bsize)
    model = LSTM_LM(train_dl.dataset.tokenizer.vocab_size + 2, model_d, hidden_d, n_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=N_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()
    writer.add_hparams({'batch_size': bsize, 'lr': lr, 'max_lr': max_lr, 'model_d': model_d, 'hidden_d': hidden_d, 'n_layers': n_layers}, {})
    
    model = train(model, train_dl, eval_dl, test_dl, optimizer, loss_fn, writer, dev, epochs=N_epochs)
    model.to(dev)

    torch.save(model.state_dict(), 'model.pth')
    
