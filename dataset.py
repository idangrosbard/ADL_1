from datasets import load_dataset
import torch
from torch.utils import data
from transformers import DataCollatorForLanguageModeling, default_data_collator
from transformers import GPT2TokenizerFast
import time
from pathlib import Path
DEBUG = False


def setup_tokenizer(use_default: bool = True):
    finetuned_tokenizer_path = Path('/content/ADL_1/wikitext-103-tokenizer-finetuned-lra')
    if use_default:
        new_tokenizer = GPT2TokenizerFast.from_pretrained("Kristijan/wikitext-103-tokenizer")
        print('init vocab size', new_tokenizer.vocab_size)
        # new_tokenizer.add_special_tokens({"mask_token": "<MASK>"})
        new_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        new_tokenizer.add_special_tokens({'eos_token': '<EOS>'})
        print('vocab size with special', new_tokenizer.vocab_size)
        print(new_tokenizer.vocab_size)
    else:
        if finetuned_tokenizer_path.exists():
            print('Found tokenizer')
            new_tokenizer = GPT2TokenizerFast.from_pretrained(str(finetuned_tokenizer_path))
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained("Kristijan/wikitext-103-tokenizer")
            tokenizer.add_special_tokens({"mask_token": "<MASK>"})
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})

            lra_path = Path(r'C:\Users\idg77\University\gylab\aclImdb\train')

            cls_generator = (f for f in lra_path.iterdir() if f.is_dir())
            
            def get_file_txt(pth: Path):
                with open(pth, 'r', encoding="utf8") as f:
                    return f.read()

            sample_generator = ((get_file_txt(sample) for sample in cls_dir.iterdir()) for cls_dir in cls_generator)
            new_tokenizer = tokenizer.train_new_from_iterator(sample_generator, 28441)
            
            # save pretrained tokenizer to disk
            new_tokenizer.save_pretrained("./wikitext-103-tokenizer-finetuned-lra")

    encoded_input = new_tokenizer('<PAD>', truncation=True, padding=False, return_tensors="pt")
    print(f"PAD: {encoded_input}")
    # encoded_input = new_tokenizer('<MASK>', truncation=True, padding=False, return_tensors="pt")
    # print(f"MASK: {encoded_input}")
    encoded_input = new_tokenizer('<EOS>', truncation=True, padding=False, return_tensors="pt")
    print(f"EOS: {encoded_input}")
    
    return new_tokenizer



class WikiTextDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = [s for s in dataset if len(s['text']) > 0]
        self.tokenizer = setup_tokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # add CLS
        samp = self.dataset[idx]['text']
        return self.tokenizer(samp)
    

class LRAClfDataset(data.Dataset):
    def __init__(self, root_path: Path):
        self.n = 0
        self.samples = []
        self.cls_map = []
        self.root_path = root_path
        for cls_folder in root_path.iterdir():
            if cls_folder.is_dir():
                self.cls_map.append(cls_folder.name)
                for sample in cls_folder.iterdir():
                    self.samples.append((sample, cls_folder.name))
                    self.n += 1
        self.tokenizer = setup_tokenizer()

            
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # add CLS
        samp, cls = self.samples[idx]
        with open(samp, 'r', encoding='utf8') as f:
            samp = f.read()
        dict = self.tokenizer(samp)
        dict['label'] = self.cls_map.index(cls)
        return dict


class LRAARDataset(data.Dataset):
    def __init__(self, root_path: Path):
        self.n = 0
        self.samples = []
        self.root_path = root_path
        for cls_folder in root_path.iterdir():
            if cls_folder.is_dir():
                self.cls_map.append(cls_folder.name)
                for sample in cls_folder.iterdir():
                    self.samples.append(sample)
                    self.n += 1
        self.tokenizer = setup_tokenizer()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # add CLS
        samp = self.samples[idx]
        with open(samp, 'r', encoding='utf8') as f:
            samp = f.read()
        return self.tokenizer(samp)


def setup_lra_clf_dataloaders(batch_size: int = 2, model: str = 'decoder'):
    t0 = time.time()
    train_ds = LRAClfDataset(Path(r'C:\Users\idg77\University\gylab\aclImdb\train'))
    test_ds = LRAClfDataset(Path(r'C:\Users\idg77\University\gylab\aclImdb\test'))

    dc = DataCollatorForLanguageModeling(tokenizer=train_ds.tokenizer, mlm=False, mlm_probability=0)

    train_dl = data.DataLoader(train_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(test_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    t1 = time.time()
    print(f'Dataloaders setup in {t1 - t0} seconds')
    return train_dl, test_dl


def setup_lra_ar_dataloaders(batch_size: int = 2, model: str = 'decoder'):
    t0 = time.time()
    train_ds = LRAARDataset(Path(r'C:\Users\idg77\University\gylab\aclImdb\train'))
    test_ds = LRAARDataset(Path(r'C:\Users\idg77\University\gylab\aclImdb\test'))
    dc = DataCollatorForLanguageModeling(tokenizer=train_ds.tokenizer, mlm=(model != 'decoder'), mlm_probability=0.15 if model != 'decoder' else 0)

    train_dl = data.DataLoader(train_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(test_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    t1 = time.time()
    print(f'Dataloaders setup in {t1 - t0} seconds')
    return train_dl, test_dl


def setup_wikitext_dataloaders(batch_size: int = 2, model: str = 'decoder'):
    t0 = time.time()
    if DEBUG:
        cache_path = Path('./cache.ds')
        if cache_path.exists():
            ds = torch.load(cache_path)
            train_ds = WikiTextDataset(ds['train'])
            eval_ds = WikiTextDataset(ds['validation'])
            test_ds = WikiTextDataset(ds['test'])
        else:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
            sample_ds = {'train': [ds['train'][i] for i in range(100)], 'validation': [ds['validation'][i] for i in range(100)], 'test': [ds['test'][i] for i in range(100)]}
            torch.save(sample_ds, cache_path)
            train_ds = WikiTextDataset(sample_ds['train'])
            eval_ds = WikiTextDataset(sample_ds['validation'])
            test_ds = WikiTextDataset(sample_ds['test'])
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

        train_ds = WikiTextDataset(ds['train'])
        eval_ds = WikiTextDataset(ds['validation'])
        test_ds = WikiTextDataset(ds['test'])

    dc = DataCollatorForLanguageModeling(tokenizer=train_ds.tokenizer, mlm=(model != 'decoder'), mlm_probability=0.15 if model != 'decoder' else 0)

    train_dl = data.DataLoader(train_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    eval_dl = data.DataLoader(eval_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(test_ds, collate_fn=dc, batch_size=batch_size, shuffle=True)
    t1 = time.time()
    print(f'Dataloaders setup in {t1 - t0} seconds')

    return train_dl, eval_dl, test_dl


def setup_dataloaders(batch_size: int = 2, dataset: str = 'wikitext', model: str = 'decoder'):
    if dataset == 'wikitext':
        return setup_wikitext_dataloaders(batch_size, model)
    elif dataset == 'lra_clf':
        return setup_lra_clf_dataloaders(batch_size, model)
    elif dataset == 'lra_ar':
        return setup_lra_ar_dataloaders(batch_size, model)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')