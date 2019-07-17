import os
import pickle
from typing import List, Callable, Tuple

import torch
from torchtext.data import Field, Dataset, Example, BucketIterator
from firelab.utils.fs_utils import load_config
from src.utils import itos_many

from src.models.transformer_lm import TransformerLM
from src.inference import InferenceState
from .utils import tokenize, apply_bpe

DEVICE = 'cuda'


def load_model(model_path:str, config_path:str, field:Field) -> TransformerLM:
    config = load_config(config_path).hp.transformer
    model = TransformerLM(config, field.vocab).to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    return model


def load_field(vocab_path:str) -> Field:
    field = Field(init_token='<bos>', eos_token='<eos>', batch_first=True, pad_first=True)
    field.vocab = pickle.load(open(vocab_path, 'rb'))

    return field


def predict(model, field, sentence_beginnings:List[str], max_len=50, batch_size=100) -> List[str]:
    examples = [Example.fromlist([x], [('text', field)]) for x in sentence_beginnings]
    dataset = Dataset(examples, [('text', field)])
    dataloader = BucketIterator(dataset, batch_size, repeat=False)
    results = []

    for batch in dataloader:
        input = batch.text.to(DEVICE)
        _, input_state = model(input[:, :-1], return_state=True)

        curr_results = InferenceState({
            "model": model.cached_forward,
            "inputs": input_state,
            "vocab": field.vocab,
            "device": DEVICE,
            "max_len": max_len,
            "is_inputs_update_enabled": True,
            "inputs_batch_dim": 1,
            "active_seqs": input,
            "sample_from_top": 0.5,
            "temperature": 0.2
        }).inference()

        curr_results = [x.cpu().numpy().tolist() for x in curr_results]
        curr_results = itos_many(curr_results, field.vocab, remove_special=True)

        results.extend(curr_results)

    return results


def build_predict_fn(model_path:str, model_config_path:str, vocab_path:str, bpes_path:str, max_len:int=50) -> Callable:
    field = load_field(vocab_path)
    model = load_model(model_path, model_config_path, field)

    def predict_fn(sentence_beginnings:List[str]) -> List[str]:
        sentence_beginnings = tokenize(sentence_beginnings)
        sentence_beginnings = apply_bpe(bpes_path, sentence_beginnings)

        return predict(
            model=model,
            field=field,
            sentence_beginnings=sentence_beginnings,
            max_len=max_len)

    return predict_fn

