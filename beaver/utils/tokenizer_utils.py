from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np


def get_tokenizer(train, vocab_size=None, unk=False):
    counts = Counter(w for ws in train["sent"] for w in ws)
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



def tokenize(sents, w_idx, max_len=None):
    unk_id = w_idx.get(UNK, 0)
    max_len = max(len(s) for s in sents)
    out = []
    for s in sents:
        t = [w_idx.get(c, unk_id) for c in s]
        if len(t) < max_len:
            t += [w_idx[PAD]] * (max_len - len(t))
        out.append(t)
    return np.stack(out, 0)

def decode(idxs, idx_w):
    out = [idx_w[idx] for idx in idxs]
    return out
