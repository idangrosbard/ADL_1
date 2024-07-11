from lstm.lstm_lm import LSTM_LM
import torch
from dataset import setup_dataloaders
import torch.utils

from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def do_batch(model, batch, optimizer, loss_fn, writer: SummaryWriter, device, train: bool = True):
    optimizer.zero_grad()
    b_inp = batch['input_ids'].to(device, non_blocking=True).long()
    labels = batch['labels'].long().to(device, non_blocking=True)

    h, c = None, None
    for t in range(1, labels.shape[1]):
        logits, h, c = model(b_inp[:, t - 1], h, c)
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
    pbar = tqdm(dataloader, desc='train' if train else 'eval')
    for batch in pbar:
        b_loss = do_batch(model, batch, optimizer, loss, writer, device, train)
        total_loss += b_loss / len(dataloader)
        pbar.set_description(f'{"train" if train else "eval"}, Loss: {b_loss}')
        writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', b_loss, global_step)
        writer.flush()
        global_step += 1
    return total_loss


def train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, loss_fn, writer: SummaryWriter, device, epochs: int = 1, eval_every: int = 1):
    for e in range(epochs):
        total_loss = do_epoch(model, train_dataloader, optimizer, loss_fn, writer, device)
        print(f'train loss: {total_loss}')
        writer.add_scalar(f'train/epoch_loss', total_loss, e)
        writer.flush()
        
        if e % eval_every == 0:
            total_loss = do_epoch(model, eval_dataloader, optimizer, loss_fn, writer, device, train=False)
            print(f'eval loss: {total_loss}')
            writer.add_scalar(f'eval/epoch_loss', total_loss, e)
            writer.flush()
    
    total_loss = do_epoch(model, test_dataloader, optimizer, loss_fn, writer, device, train=False)
    print(f'Test loss: {total_loss}')
    writer.add_scalar(f'test/epoch_loss', total_loss, e)
    writer.flush()
    return model

if __name__ == '__main__':
    global_step = 0
    N_epochs = 10
    bsize = 64
    lr = 1e-6
    max_lr = 1e-4
    d_input = 16
    hidden_d = 64
    n_layers = 3
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl, eval_dl, test_dl = setup_dataloaders(bsize, 'wikitext', 'lstm')
    model = LSTM_LM(train_dl.dataset.tokenizer.vocab_size + 2, hidden_d, n_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=N_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()
    writer.add_hparams({'batch_size': bsize, 'lr': lr, 'max_lr': max_lr, 'd_input': d_input, 'hidden_d': hidden_d, 'n_layers': n_layers}, {})
    
    model.to(dev)
    model = train(model, train_dl, eval_dl, test_dl, optimizer, loss_fn, writer, dev, epochs=N_epochs)
    

    torch.save(model.state_dict(), 'model.pth')
    
