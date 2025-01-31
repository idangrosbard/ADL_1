import torch
import torch.utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformer import EncoderLLM
from dataset import setup_dataloaders
from ssm import S4Model
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='s4', choices=['s4', 'transformer'])

    return parser.parse_args()



def do_batch(model, batch, optimizer, loss_fn, writer: SummaryWriter, device, train: bool = True, transformer: bool = True):
    optimizer.zero_grad()
    b_inp = batch['input_ids'].to(device, non_blocking=True).long()
    labels = batch['labels'].long()
    labels = labels.to(device, non_blocking=True)
    attn = batch['attention_mask'].to(device, non_blocking=True)

    if transformer:
        logits = model(b_inp, attn)
    else:
        logits = model(b_inp)


    if False:
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
    if True:
        # labels of shape [b, l]
        # from each batch gather only the logits that correspond to the labels that are not -100
        logits = logits[labels != -100]
        labels = labels[labels != -100]

    loss = loss_fn(logits, labels)
    
    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    # writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', loss.item())
    # writer.flush()
    return loss.item()
    

def do_epoch(model, dataloader, optimizer, loss, writer: SummaryWriter, device, train: bool = True, transformer: bool = True, global_step: int = 0):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    pbar = tqdm(dataloader, desc='train' if train else 'eval')
    for batch in pbar:
        batch_loss = do_batch(model, batch, optimizer, loss, writer, device, train, transformer)
        total_loss += batch_loss / len(dataloader)
        pbar.set_description(f'{"train" if train else "eval"}, Loss: {batch_loss}')
        
        writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', batch_loss, global_step)
        writer.flush()
        global_step += 1
    
    
    return total_loss, global_step


def train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, loss_fn, writer: SummaryWriter, device, epochs: int = 1, eval_every: int = 1, transformer: bool = True):
    global_step = 0
    for e in range(epochs):
        total_loss, global_step = do_epoch(model, train_dataloader, optimizer, loss_fn, writer, device, transformer=transformer, global_step=global_step)
        print(f'train loss: {total_loss}')
        writer.add_scalar(f'train/epoch_loss', total_loss, e)
        writer.flush()
        
        if e % eval_every == 0:
            total_loss, global_step = do_epoch(model, eval_dataloader, optimizer, loss_fn, writer, device, train=False, transformer=transformer, global_step=global_step)
            writer.add_scalar(f'val/epoch_loss', total_loss, e)
            writer.flush()
            print(f'eval loss: {total_loss}')
    
    total_loss, global_step = do_epoch(model, test_dataloader, optimizer, loss_fn, writer, device, train=False, transformer=transformer, global_step=global_step)
    writer.add_scalar(f'test/epoch_loss', total_loss, e)
    writer.flush()
    print(f'Test loss: {total_loss}')
    return model



def get_transformer_llm(vocab_size, writer: SummaryWriter = None):
    model_d = 128
    attn_d = 64
    n_heads = 4
    ff_d = 256
    n_layers = 4
    writer.add_hparams({'model_d': model_d, 'attn_d': attn_d, 'n_heads': n_heads, 'ff_d': ff_d, 'n_layers': n_layers}, {})
    model = EncoderLLM(vocab_size + 2, model_d, attn_d, ff_d, n_heads, n_layers)
    return model


def get_s4_llm(vocab_size, writer: SummaryWriter = None):
    
    H = 256
    N = 16
    n_layers = 3
    writer.add_hparams({'H': H, 'N': N, 'n_layers': n_layers}, {})
    model = S4Model(H, N, vocab_size + 2, vocab_size + 2, n_layers)
    return model

if __name__ == '__main__':
    
    args = get_args()
    N_epochs = args.epochs
    bsize = args.bsize
    lr = args.lr
    max_lr = args.max_lr
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer=SummaryWriter()

    train_dl, eval_dl, test_dl = setup_dataloaders(bsize, 'wikitext', 's4')
    if args.model == 'transformer':
        model = get_transformer_llm(train_dl.dataset.tokenizer.vocab_size, writer)
    else:
        model = get_s4_llm(train_dl.dataset.tokenizer.vocab_size, writer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # add gradient clipping:
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=N_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    writer.add_hparams({'batch_size': bsize, 'lr': lr, 'max_lr': max_lr}, {})
    
    model.to(dev)
    model = train(model, train_dl, eval_dl, test_dl, optimizer, loss_fn, writer, dev, epochs=N_epochs, transformer=(type(model) == EncoderLLM))
    

    torch.save(model.state_dict(), 'model.pth')
    

