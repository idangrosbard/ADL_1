import torch
import torch.utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformer import EncoderLLM, EncoderClf
from dataset import setup_dataloaders
from ssm import S4Model, S4Classifier


def do_batch(model, batch, optimizer, loss_fn, writer: SummaryWriter, device, train: bool = True, transformer: bool = True):
    optimizer.zero_grad()
    x = batch['input_ids'].to(device).long()
    y = batch['label'].long()
    attn = batch['attention_mask']
    L = attn.sum(dim=1)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    if transformer:
        logits = model(x, attn)
    else:
        logits = model(x, L)

    loss = loss_fn(logits, y)
    
    if train:
        loss.backward()
        optimizer.step()
    writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', loss.item())
    writer.flush()
    return loss.item()
    

def do_epoch(model, dataloader, optimizer, loss, writer: SummaryWriter, device, train: bool = True, transformer: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    for batch in tqdm(dataloader, desc='train' if train else 'eval'):
        total_loss += do_batch(model, batch, optimizer, loss, writer, device, train, transformer) / len(dataloader)
    writer.add_scalar(f'{"train" if train else "eval"}/epoch_loss', total_loss)
    writer.flush()
    return total_loss


def train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, loss_fn, writer: SummaryWriter, device, epochs: int = 1, eval_every: int = 1, transformer: bool = True):
    for e in range(epochs):
        total_loss = do_epoch(model, train_dataloader, optimizer, loss_fn, writer, device, transformer=transformer)
        print(f'train loss: {total_loss}')
        
        if e % eval_every == 0:
            total_loss = do_epoch(model, eval_dataloader, optimizer, loss_fn, writer, device, train=False, transformer=transformer)
            print(f'eval loss: {total_loss}')
    
    total_loss = do_epoch(model, test_dataloader, optimizer, loss_fn, writer, device, train=False, transformer=transformer)
    print(f'Test loss: {total_loss}')
    return model



def get_transformer_llm(vocab_size, writer: SummaryWriter = None):
    model_d = 16
    attn_d = 64
    n_heads = 8
    ff_d = 256
    n_layers = 3
    writer.add_hparams({'model_d': model_d, 'attn_d': attn_d, 'n_heads': n_heads, 'ff_d': ff_d, 'n_layers': n_layers}, {})
    model = EncoderClf(EncoderLLM(vocab_size + 2, model_d, attn_d, ff_d, n_heads, n_layers), model_d, 3)
    return model


def get_s4_llm(vocab_size, writer: SummaryWriter = None):
    
    H = 16
    N = 64
    n_layers = 3
    writer.add_hparams({'H': H, 'N': N, 'n_layers': n_layers}, {})
    model = S4Classifier(S4Model(H, N, vocab_size + 2, vocab_size + 2, n_layers), H, 3)
    return model

if __name__ == '__main__':
    N_epochs = 10
    bsize = 2
    lr = 1e-5
    max_lr = 1e-4
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer=SummaryWriter()

    train_dl, test_dl = setup_dataloaders(bsize, 'lra_clf')
    model = get_transformer_llm(train_dl.dataset.tokenizer.vocab_size, writer)
    model = get_s4_llm(train_dl.dataset.tokenizer.vocab_size, writer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=N_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    writer.add_hparams({'batch_size': bsize, 'lr': lr, 'max_lr': max_lr}, {})
    
    model = train(model, train_dl, test_dl, test_dl, optimizer, loss_fn, writer, dev, epochs=N_epochs, transformer=(type(model) == EncoderLLM))
    model.to(dev)

    torch.save(model.state_dict(), 'model.pth')
    
