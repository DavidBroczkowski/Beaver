
from collections import Counter
import copy
from copy import deepcopy
import itertools
import math
from pathlib import Path
import random
import re
import string

import gensim.downloader
import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
import torch
from torch import nn



class LocalGlove:
    """
    A class containing methods for loading GLoVE word embeddings, implemented from Transformer Programs by Friedman et al.

    Attributes:
        - key_to_index: a dictionary that maps words to the index they appear in in the embedding file
        - vectors: a NumPy array containing the vectors for each word indexed. 
    """
    def __init__(self, fn, idx_w=None):
        """
        Initializes the class and its attributes

        Inputs:
            - fn: a String containing the path of the GLoVE embeddings file, in a .txt format
            - idx_w: a NumPy array where the input of an index returns a word. Maps indices to words
        """
        rows = []
        self.key_to_index = {}
        need = set(idx_w) if idx_w is not None else None
        with open(fn, "r", encoding='utf-8') as f:
            for line in f:
                i = line.find(" ")
                w = line[:i]
                if (not need) or w in need:
                    parts = line.strip().split(" ")
                    self.key_to_index[parts[0]] = len(rows)
                    rows.append(np.array([float(v) for v in parts[1:]]))
        self.vectors = np.stack(rows, 0)
        #logger.info(f"loaded {len(self.vectors)} rows from {fn}")


def get_glove_embeddings(
    idx_w,
    name="glove-wiki-gigaword-100",
    dim=None,
):
    """
    Retrieves the GLoVE embeddings

    Inputs: 
        - idx_w: a NumPy array where the input of an index returns a word. Maps indices to words
        - name: Can be either the path of the GLoVE embeddings .txt file, begins with "data", or the name of the GLoVE file to download

    Output:
        - a 2d NumPy array containing the GLoVE embeddings for the given idx_w by the token's id
    """
    if name.startswith("data"):
        glove_vectors = LocalGlove(name, idx_w)
    else:
        glove_vectors = gensim.downloader.load(name)
    lst = []
    V = glove_vectors.vectors
    missing = []
    for w_ in idx_w:
        if name.startswith("data"):
            w = w_
        else:
            w = w_.lower()
        if w in glove_vectors.key_to_index:
            lst.append(V[glove_vectors.key_to_index[w]])
        else:
            lst.append(np.random.randn(V.shape[1]))
            missing.append(w)
    #logger.info(f"found {len(lst)-len(missing)}/{len(lst)} glove embeddings")
    #logger.info(f"missing {missing[:10] + ['...']}")
    emb = np.stack(lst, 0)
    if dim is not None and dim != emb.shape[-1]:
        emb = GaussianRandomProjection(
            n_components=dim, random_state=0
        ).fit_transform(emb)
    return emb

def embed(doc, idx_emb):
    """
    Embeds a document, doc, using the embeddings given by idx_emb

    Inputs:
        - doc: an array containing the tokenized document, where each element is the ID of a token
        - idx_emb: a 2D NumPy array containing indexed embedding vectors
    Output:
        - a torch.Tensor of size (D, E), where D is |doc| and E is the siye of the embedding vectors, 
          containing the embedding vectors for the document's tokens
    """
    if not isinstance(doc, list):
        print("[ERROR] To embed a document, it must be in the form of a list of tokens")
        return
    return torch.stack(idx_emb[doc])