from collections import Counter
from nltk.tokenize import word_tokenize
from llguidance import TokenizerWrapper, LLTokenizer
import numpy as np

BOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
PAD = "<pad>"
UNK = "<unk>"


class NLTK_Tokenizer:
    """
    A class that acts as a tokenizer using the NLTK standard, implemented from Transformer Programs by Friedman et al.

    Attributes:
        - idx_w: a NumPy array where the input of an index returns a word. Maps indices to words
        - w_idx: a dictionary mapping words to their indices.
    """
    def __init__(self, train):
        """
        Initializes the class and its attributes

        Inputs:
            - train: a list[dict] containing the training data. 
                     Each dict must have a "prompt" key structured as a list of tokenized or separated words
        """

        # initialize tokenizer
        idx_w, w_idx = self.get_tokenizer(train)
        self.idx_w = idx_w
        self.w_idx = w_idx

        return

    def get_tokenizer(self, train, vocab_size=None, unk=False):
        """
        Retrieves the mappings between indices and words

        Inputs:
            - train: a list[dict] containing the training data. 
                        Each dict must have a "prompt" key structured as a list of tokenized or separated words
            - vocab_size: an integer that remaps the input vocabulary to this integer, taking the most common words first
            - unk: when True, appends an additional UNK token whenever PAD is added
        Outputs:
            - idx_w: a NumPy array where the input of an index returns a word. Maps indices to words
            - w_idx: a dictionary mapping words to their indices.
        """
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
        """
        Maps word tokens into their respective ids and pads the sentences to max_length if specified

        Inputs:
            - sents: a 2d list containing documents and their word tokens
            - max_len: an integer containing the maximum length of the documents. If not specified, is set to the length of longest document
        Outputs:
            - a 2d NumPy array containing the documents converted to ids and padded
        """
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

def initialize_llguidance(w_idx, idx_w):
    """
    Creates an LLTokenizer object from w_idx and idx_w using the NLTK tokenizer, 
    takes the place of from_tokenizer() used with HuggingFace

    Input:
        - idx_w: a NumPy array where the input of an index returns a word. Maps indices to words
        - w_idx: a dictionary mapping words to their indices.
    Output:
        - an LLTokenizer object
    """
    def gtokenizer(text):
        words = word_tokenize(text.decode("utf-8") if isinstance(text, bytes) else text)
        return [w_idx.get(w, w_idx[UNK]) for w in words]

    gtokenizer.eos_token_id = w_idx[EOS]
    gtokenizer.bos_token_id = w_idx[BOS]
    gtokenizer.tokens = [w.encode("utf-8") for w in idx_w]
    gtokenizer.special_token_ids = [w_idx[t] for t in [PAD, BOS, EOS] if t in w_idx]

    twrapper = TokenizerWrapper(gtokenizer)
    lltokenizer = LLTokenizer(twrapper)

    return lltokenizer