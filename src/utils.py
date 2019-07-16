import math
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm


SPECIAL_TOKENS = {'<bos>', '<eos>', '<pad>', '<unk>'}


def itos_many(seqs:List[List[int]], vocab, remove_special=True, sep=' '):
    """
    Converts sequences of token ids to normal strings
    """
    sents = [[vocab.itos[i] for i in seq] for seq in seqs]

    if remove_special:
        sents = remove_special_symbols(sents)

    sents = [sep.join(s).replace('@@ ', '') for s in sents]

    return sents


def remove_special_symbols(sentences):
    return [[t for t in s if not t in SPECIAL_TOKENS] for s in sentences]
