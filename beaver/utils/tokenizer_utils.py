from collections import Counter
from nltk.tokenize import word_tokenize
from llguidance import TokenizerWrapper
import numpy as np

BOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
PAD = "<pad>"
UNK = "<unk>"


class NLTK_Tokenizer:
    def __init__(self, train):

        # initialize tokenizer
        idx_w, w_idx = self.get_tokenizer(train)
        self.idx_w = idx_w
        self.w_idx = w_idx

        return

    def get_tokenizer(self, train, vocab_size=None, unk=False):
        counts = Counter(w for row in train for w in row["prompt"])
        words = []
        for w in [PAD] + ([UNK] if unk else []):
            words.append(w)
        if vocab_size:
            words += [w for w, _ in counts.most_common() if w not in words][
                :vocab_size
            ]
        else:
            words += sorted(c for c in counts.keys() if c not in words)
        idx_w = np.array(words)
        w_idx = {w: i for i, w in enumerate(idx_w)}

        return idx_w, w_idx

    def tokenize(self, sents, max_len=None):
        unk_id = self.w_idx.get(UNK, 0)
        max_len = max(len(s) for s in sents)
        out = []
        for s in sents:
            t = [self.w_idx.get(c, unk_id) for c in s]
            if len(t) < max_len:
                t += [self.w_idx[PAD]] * (max_len - len(t))
            out.append(t)
        return np.stack(out, 0)

    def decode(self, idxs):
        out = [self.idx_w[idx] for idx in idxs]
        return out
    

def normalize_sent(sent):
    """
    Appends and prepends SOS and EOS tokens respectfully and tokenizes the sentence via the NLTK schema

    Input:
        - sent: a String containing the sentence to normalize
    Output:
        - a list containing the separated normalized sentence
    """
    return ["<s>"] + word_tokenize(sent) + ["</s>"]

def initialize_llguidance(w_idx, Tokenizer: Any):
    """
    Creates an LLTokenizer object from the w_idx conversion array using the NLTK tokenizer, 
    takes the place of from_tokenizer() used with HuggingFace

    Input:
        - w_idx: a dictionary where words are the keys and their indices are the values
    Output:
        - an LLTokenizer object
    """
    def gtokenizer(text):
        words = word_tokenize(text.decode("utf-8") if isinstance(text, bytes) else text)
        return [w_idx.get(w, unk_id) for w in words]

    gtokenizer.eos_token_id = w_idx[EOS]
    gtokenizer.bos_token_id = w_idx[BOS]
    gtokenizer.tokens = [w.encode("utf-8") for w in idx_w]
    gtokenizer.special_token_ids = [w_idx[t] for t in [PAD, BOS, EOS] if t in w_idx]

    twrapper = TokenizerWrapper(gtokenizer)
    lltokenizer = LLTokenizer(twrapper)

    return lltokenizer