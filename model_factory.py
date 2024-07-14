import torch
from torch.utils.tensorboard import SummaryWriter
from lstm import LSTM_LM, LSTMClassifier
from transformer import Decoder, DecoderClf
from ssm import S4Model, S4Classifier
from typing import Optional
from pathlib import Path



def get_transformer_llm(vocab_size, writer: SummaryWriter = None):
    model_d = 512
    n_heads = 8
    attn_d = model_d // n_heads
    ff_d = 2048
    n_layers = 4
    writer.add_hparams({'model_d': model_d, 'attn_d': attn_d, 'n_heads': n_heads, 'ff_d': ff_d, 'n_layers': n_layers}, {})
    model = Decoder(vocab_size + 2, model_d, attn_d, ff_d, n_heads, n_layers, vocab_size + 2, decoder_only=True, return_logits=True)
    return model


def get_lstm_llm(vocab_size, writer: SummaryWriter = None):
    d_input = 256
    hidden_d = 256
    n_layers = 4
    writer.add_hparams({'d_input': d_input, 'hidden_d': hidden_d, 'n_layers': n_layers}, {})
    model = LSTM_LM(vocab_size + 2, hidden_d, n_layers)
    return model


def get_s4_llm(vocab_size, writer: SummaryWriter = None):
    H = 256
    N = 16
    n_layers = 4
    writer.add_hparams({'H': H, 'N': N, 'n_layers': n_layers}, {})
    model = S4Model(H, N, vocab_size + 2, vocab_size + 2, n_layers)
    return model


def get_model(model_type: str, is_llm: bool, vocab_size: int, writer: SummaryWriter = None, n_classes: int = 3, pretrained_weights: Optional[Path] = None):
    if model_type == 'transformer':
        llm = get_transformer_llm(vocab_size, writer)
    elif model_type == 'lstm':
        llm = get_lstm_llm(vocab_size, writer)
    elif model_type == 's4':
        llm = get_s4_llm(vocab_size, writer)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, map_location = torch.device('cpu'))
        llm.load_state_dict(state_dict)
    
    if is_llm:
        if model_type == 'transformer':
            return DecoderClf(llm, llm.d_model, n_classes)
        elif model_type == 'lstm':
            return LSTMClassifier(llm, llm.d_hidden, n_classes)
        elif model_type == 's4':
            return S4Classifier(llm, llm.H, n_classes)
    else:
        return llm
