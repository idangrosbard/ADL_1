from lstm.lstm_lm import LSTM_LM
from ssm import S4Model
from transformer.decoder import Decoder
import torch
from dataset import setup_dataloaders
import torch.utils

from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse


def do_batch(model, batch, optimizer, loss_fn, writer: SummaryWriter, device, train: bool = True):
    optimizer.zero_grad()
    b_inp = batch['input_ids'].to(device, non_blocking=True).long()

    batch_losses = torch.zeros(b_inp.shape[1] - 1)

    if type(model) == LSTM_LM:
        for t in range(1, b_inp.shape[1]):
            h, c = None, None
            for t_tag in range(0, t):
                logits, h, c = model(b_inp[:, t_tag], h, c)
            loss = loss_fn(logits, b_inp[:, t])
            if train:
                loss.backward()
                optimizer.step()
            batch_losses[t] = loss.item()

    elif type(model) == Decoder:
        mask = batch['attention_mask'].to(device, non_blocking=True)
        for t in range(1, b_inp.shape[1] - 1):
            # Get the logits for all the tokens in the sequence, up to length t
            logits = model(b_inp[:, :t], None, mask=mask[:, :t])
            # Get only the logits for the last token in the sequence
            logits = logits[:, -1, :]
            # Calc loss, based on 0:t tokens, predict the t+1 token
            
            loss = loss_fn(logits, b_inp[:, t + 1])
            if train:
                loss.backward()
                optimizer.step()
            print(loss.item())
            batch_losses[t] = (loss.item())
    
    elif type(model) == S4Model:
        mask = batch['attention_mask'].to(device, non_blocking=True)
        # Get the logits for all the tokens in the sequence
        logits = model(b_inp, mask=mask)
        # Calc all loses for all tokens in the sequence
        # Based on 0:t tokens, predict the t+1 token
        loss = loss_fn(logits[:-1], b_inp[1:])
        if train:
            loss.backward()
            optimizer.step()
        batch_losses[t] = (loss.item())

    return torch.tensor(batch_losses).mean()
    
    
def do_epoch(model, dataloader, optimizer, loss, writer: SummaryWriter, device, train: bool = True, global_step: int = 0):
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
    return total_loss, global_step


def train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, loss_fn, writer: SummaryWriter, device, epochs: int = 1, eval_every: int = 1):
    global_step = 0
    for e in range(epochs):
        total_loss, global_step = do_epoch(model, train_dataloader, optimizer, loss_fn, writer, device, global_step=global_step)
        print(f'train loss: {total_loss}')
        writer.add_scalar(f'train/epoch_loss', total_loss, e)
        writer.flush()
        
        if e % eval_every == 0:
            total_loss, global_step = do_epoch(model, eval_dataloader, optimizer, loss_fn, writer, device, train=False, global_step=global_step)
            print(f'eval loss: {total_loss}')
            writer.add_scalar(f'eval/epoch_loss', total_loss, e)
            writer.flush()
    
    total_loss, global_step = do_epoch(model, test_dataloader, optimizer, loss_fn, writer, device, train=False, global_step=global_step)
    print(f'Test loss: {total_loss}')
    writer.add_scalar(f'test/epoch_loss', total_loss, e)
    writer.flush()
    return model


def get_transformer_llm(vocab_size, writer: SummaryWriter = None):
    model_d = 128
    attn_d = 64
    n_heads = 8
    ff_d = 256
    n_layers = 6
    writer.add_hparams({'model_d': model_d, 'attn_d': attn_d, 'n_heads': n_heads, 'ff_d': ff_d, 'n_layers': n_layers}, {})
    model = Decoder(vocab_size + 2, model_d, attn_d, ff_d, n_heads, n_layers, vocab_size + 2, decoder_only=True, return_logits=True)
    return model


def get_lstm_llm(vocab_size, writer: SummaryWriter = None):
    d_input = 256
    hidden_d = 256
    n_layers = 6
    writer.add_hparams({'d_input': d_input, 'hidden_d': hidden_d, 'n_layers': n_layers}, {})
    model = LSTM_LM(vocab_size + 2, hidden_d, n_layers)
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['transformer', 'lstm', 's4'], default='lstm')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    N_epochs = 10
    bsize = 64
    lr = 1e-5
    max_lr = 1e-3
    d_input = 16
    hidden_d = 64
    n_layers = 3
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    train_dl, eval_dl, test_dl = setup_dataloaders(bsize, 'wikitext', 'lstm')
    # model = LSTM_LM(train_dl.dataset.tokenizer.vocab_size + 2, hidden_d, n_layers)
    if args.model_type == 'transformer':
        model = get_transformer_llm(train_dl.dataset.tokenizer.vocab_size, writer)
    elif args.model_type == 'lstm':
        model = get_lstm_llm(train_dl.dataset.tokenizer.vocab_size, writer)
    else:
        raise ValueError('Model type not supported')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=N_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    writer.add_hparams({'batch_size': bsize, 'lr': lr, 'max_lr': max_lr, 'd_input': d_input, 'hidden_d': hidden_d, 'n_layers': n_layers}, {})
    
    model.to(dev)
    model = train(model, train_dl, eval_dl, test_dl, optimizer, loss_fn, writer, dev, epochs=N_epochs)

    torch.save(model.state_dict(), 'model.pth')
    
