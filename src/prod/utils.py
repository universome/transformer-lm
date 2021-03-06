from subprocess import check_output
from typing import List

import nltk; nltk.download('punkt')


def apply_bpe(bpes_path:str, sentences:List[str]) -> List[str]:
    """
        Wrapper for BPE function from subword-nmt:
        $ subword-nmt apply-bpe -c $bpes_path < $dataset > $dataset.bpe
    """
    cmd = ['subword-nmt', 'apply-bpe', '-c', bpes_path]
    cmd_input = '\n'.join(sentences)
    results = check_output(cmd, input=cmd_input, encoding='utf8')
    results = results.splitlines()

    return results


def tokenize(sentences:List[str]) -> List[str]:
    return [' '.join(nltk.word_tokenize(s)) for s in sentences]
