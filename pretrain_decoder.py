from lstm import LSTM_LM
from ssm import S4Model
from transformer import Decoder
import torch
from dataset import setup_dataloaders
import torch.utils

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from model_factory import get_model
from pathlib import Path

def do_batch(model, batch, optimizer, loss_fn, writer: SummaryWriter, device, train: bool = True):
    optimizer.zero_grad()
    b_inp = batch['input_ids'].to(device, non_blocking=True).long()
    mask = batch['attention_mask'].to(device, non_blocking=True)

    batch_losses = torch.zeros(b_inp.shape[1] - 1)
    accs = torch.zeros(b_inp.shape[1] - 1)

    if type(model) == LSTM_LM:
        # randomly sample some subset of target tokens
        subset = 10
        subset_ts = torch.randint(1, b_inp.shape[1], subset)
        for t in subset_ts:
            h, c = None, None
            for t_tag in range(0, t):
                logits, h, c = model(b_inp[:, t_tag], h, c)
            loss = loss_fn(logits, b_inp[:, t])
            acc = (logits.argmax(dim=1) == b_inp[:, t]).float().mean()

            if train:
                loss.backward()
                optimizer.step()
            print(loss.item())
            batch_losses[t] = loss.item()
            accs[t] = acc.item()

    elif type(model) == Decoder:
        if False:
            for t in range(1, b_inp.shape[1]):
                # Get the logits for all the tokens in the sequence, up to length t (exclusive)
                logits = model(b_inp[:, :t], None, mask=mask[:, :t])
                
                # Get only the logits for the last token in the input sequence
                logits = logits[:, -1, :]

                # Calc loss, based on 0:t (exclusive) tokens, predict the t token
                loss = loss_fn(logits, b_inp[:, t])
                # assert not nan
                assert loss != torch.nan
                if train:
                    loss.backward()
                    optimizer.step()
                print(loss.item())
                batch_losses[t] = (loss.item())
        else:
            # Get the logits for all the tokens in the sequence, up to length t (exclusive)
            T = b_inp.shape[1]
            logits = model(b_inp[:, :T], None, mask=mask[:, 1:])
            logits = logits.reshape(-1, logits.shape[-1])
            target = b_inp[:, 1:].reshape(-1)
            # Calc loss, based on 0:t (exclusive) tokens, predict the t token
            loss = loss_fn(logits, target)
            # assert not nan
            assert loss != torch.nan
            if train:
                loss.backward()
                optimizer.step()
            acc = (logits.argmax(dim=1) == target).float().mean()
            return loss.item(), acc.item()
    
    elif type(model) == S4Model:
        mask = batch['attention_mask'].to(device, non_blocking=True)
        # Get the logits for all the tokens in the sequence
        logits = model(b_inp)
        # Calc all loses for all tokens in the sequence
        # Based on 0:t tokens, predict the t+1 token
        
        logits = logits[:, :-1]
        target = b_inp[:, 1:]
        logits = logits.reshape(-1, logits.shape[-1])
        loss = loss_fn(logits, target.reshape(-1))
        if train:
            loss.backward()
            optimizer.step()
        if torch.isnan(loss):
            raise ValueError('Loss is nan')
        acc = (logits.argmax(dim=1) == target.reshape(-1)).float().mean()
        return loss.item(), acc.item()
    
    return torch.tensor(batch_losses).mean(), torch.tensor(accs).mean()
    
    
def do_epoch(model, dataloader, optimizer, loss, writer: SummaryWriter, device, train: bool = True, global_step: int = 0):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_acc = 0
    pbar = tqdm(dataloader, desc='train' if train else 'eval')
    for batch in pbar:
        b_loss, b_acc = do_batch(model, batch, optimizer, loss, writer, device, train)
        total_loss += b_loss / len(dataloader)
        total_acc += b_acc / len(dataloader)
        pbar.set_description(f'{"train" if train else "eval"}, Loss: {b_loss}, Acc: {b_acc}')
        writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', b_loss, global_step)
        writer.add_scalar(f'{"train" if train else "eval"}/batch_acc', b_acc, global_step)
        writer.flush()
        global_step += 1
    return total_loss, total_acc, global_step


def train_model(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, loss_fn, writer: SummaryWriter, device, epochs: int = 1, eval_every: int = 1):
    global_step = 0
    for e in range(epochs):
        model.train()
        total_loss, total_acc, global_step = do_epoch(model, train_dataloader, optimizer, loss_fn, writer, device, train=True, global_step=global_step)
        print(f'train loss: {total_loss}, acc: {total_acc}')
        writer.add_scalar(f'train/epoch_loss', total_loss, e)
        writer.add_scalar(f'train/epoch_acc', total_acc, e)
        writer.flush()
        
        
        if e % eval_every == 0:
            with torch.no_grad():
                model.eval()
                total_loss, total_acc, global_step = do_epoch(model, eval_dataloader, optimizer, loss_fn, writer, device, train=False, global_step=global_step)
                print(f'eval acc: {total_acc}')
                writer.add_scalar(f'eval/epoch_loss', total_loss, e)
                writer.add_scalar(f'eval/epoch_acc', total_acc, e)
                writer.flush()
    
    with torch.no_grad():
        model.eval()
        total_loss, total_acc, global_step = do_epoch(model, test_dataloader, optimizer, loss_fn, writer, device, train=False, global_step=global_step)
        print(f'Test loss: {total_loss}, acc: {total_acc}')
        writer.add_scalar(f'test/epoch_loss', total_loss, e)
        writer.add_scalar(f'test/epoch_acc', total_acc, e)
        writer.flush()
        return model
        




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['transformer', 'lstm', 's4'], default='transformer')
    parser.add_argument('--dataset', type=str, choices=['lra', 'wikitext'], default='wikitext')
    parser.add_argument('--pretrained_weights', type=Path, default=None)
    parser.add_argument('--weights_output_path', type=Path, default='.')
    parser.add_argument('--logdir', type=Path, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    N_epochs = 2

    bsize = 16
    if args.model_type == 'transformer':
        bsize = 2
    lr = 1e-5
    max_lr = 1e-3
    d_input = 16
    hidden_d = 64
    n_layers = 3
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=args.logdir / 'runs' / f'{args.model_type}_{args.dataset}')

    train_dl, eval_dl, test_dl = setup_dataloaders(bsize, args.dataset, 'decoder')

    model = get_model(args.model_type, True, train_dl.dataset.tokenizer.vocab_size, writer, pretrained_weights=args.pretrained_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=N_epochs)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=28439) # ignore the padding index
    
    writer.add_hparams({'batch_size': bsize, 'lr': lr, 'max_lr': max_lr, 'd_input': d_input, 'hidden_d': hidden_d, 'n_layers': n_layers, 'model_type': args.model_type, 'dataset': args.dataset}, {})
    
    model.to(dev)
    model = train_model(model, train_dl, eval_dl, test_dl, optimizer, loss_fn, writer, dev, epochs=N_epochs)

    weights_output_path = Path(args.weights_output_path) / f'{args.model_type}_{args.dataset}_lm.pth'
    weights_output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), weights_output_path)
    
