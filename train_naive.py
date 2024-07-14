import torch
import torch.utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import setup_dataloaders
from model_factory import get_model
from argparse import ArgumentParser
from typing import Tuple, Optional
from pathlib import Path



def do_batch(model, batch, optimizer, loss_fn, writer: SummaryWriter, device, train: bool = True) -> Tuple[float, float]:
    optimizer.zero_grad()
    x = batch['input_ids'].to(device, non_blocking=True).long()
    y = batch['label'].to(device, non_blocking=True).long()
    attn = batch['attention_mask'].to(device, non_blocking=True).float()
    print(attn.shape)

    logits = model(x, attn)
    loss = loss_fn(logits, y)
    
    if train:
        loss.backward()
        optimizer.step()
    
    # Calculate the accuracy:
    acc = (logits.argmax(dim=1) == y).float().mean()
    return loss.item(), acc.item()
    

def do_epoch(model, dataloader, optimizer, loss, writer: SummaryWriter, device, train: bool = True, global_steps: int = 0) -> Tuple[float, float, int]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_acc = 0
    pbar = tqdm(dataloader, desc='train' if train else 'eval')
    for batch in pbar:
        batch_loss, batch_acc = do_batch(model, batch, optimizer, loss, writer, device, train)
        total_loss += batch_loss  / len(dataloader)
        total_acc += batch_acc / len(dataloader)
        pbar.set_description(f'{"train" if train else "eval"}, Loss: {batch_loss}, acc: {batch_acc}')
        
        writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', batch_loss, global_steps)
        writer.add_scalar(f'{"train" if train else "eval"}/batch_acc', batch_acc, global_steps)
        writer.flush()
        global_steps += 1

    return total_loss, total_acc, global_steps


def train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, loss_fn, writer: SummaryWriter, device, epochs: int = 1, eval_every: int = 1):
    global_steps = 0
    for e in range(epochs):
        model.train()
        total_loss, total_acc, global_steps = do_epoch(model, train_dataloader, optimizer, loss_fn, writer, device, global_steps=global_steps)
        print(f'train loss: {total_loss}')
        print('train acc:', total_acc)
        writer.add_scalar(f'{"train" if train else "eval"}/epoch_loss', total_loss, e)
        writer.add_scalar(f'{"train" if train else "eval"}/epoch_acc', total_acc, e)
        writer.flush()
        
        
        if e % eval_every == 0:
            with torch.no_grad():
                model.eval()
                total_loss, total_acc, global_steps = do_epoch(model, eval_dataloader, optimizer, loss_fn, writer, device, train=False, global_steps=global_steps)
                print(f'eval loss: {total_loss}')
                print('eval acc:', total_acc)
                writer.add_scalar(f'{"train" if train else "eval"}/epoch_loss', total_loss, e)
                writer.add_scalar(f'{"train" if train else "eval"}/epoch_acc', total_acc, e)
                writer.flush()

    with torch.no_grad():
        model.eval()
        total_loss, total_acc, global_steps = do_epoch(model, test_dataloader, optimizer, loss_fn, writer, device, train=False, global_steps=global_steps)
        print(f'Test loss: {total_loss}')
        print('Test acc:', total_acc)
        writer.add_scalar(f'test/epoch_loss', total_loss, e)
        writer.add_scalar(f'test/epoch_acc', total_acc, e)
        writer.flush()
    return model



def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['transformer', 'lstm', 's4'], default='transformer')
    parser.add_argument('--pretrained_weights', type=Path, default=None)
    parser.add_argument('--weights_output_path', type=Path, default='.')
    parser.add_argument('--logdir', type=Path, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    N_epochs = 10
    bsize = 256
    if args.model_type == 'transformer':
        bsize = 16
    lr = 1e-5
    max_lr = 1e-3
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(dev)
    # writer = SummaryWriter(log_dir=args.logdir)
    writer = SummaryWriter(log_dir=args.logdir / 'runs' / f'{args.model_type}')

    train_dl, test_dl = setup_dataloaders(bsize, 'lra_clf')
    model = get_model(args.model_type, False, train_dl.dataset.tokenizer.vocab_size, writer, 2, pretrained_weights=args.pretrained_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=N_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    writer.add_hparams({'batch_size': bsize, 'lr': lr, 'max_lr': max_lr}, {})
    
    model.to(dev)
    loss_fn.to(dev)
    model = train(model, train_dl, test_dl, test_dl, optimizer, loss_fn, writer, dev, epochs=N_epochs)
    

    weights_output_path = Path(args.weights_output_path) / f'{args.model_type}_clf.pth'
    weights_output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), weights_output_path)
    

