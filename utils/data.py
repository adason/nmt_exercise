""" Common Data Loading methods for Languag Models.
"""
import collections

import numpy as np


def build_vocab(filename, limit=100):
    """ Build a vocabulary from the corpus.

    Parameters
    ----------
    filename : str
        Filename of the input data
    limit : int
        Limit the number of words in the vocabulary.

    Returns
    -------
    dict
        Python dictionary of {<word>: <index>} pairs.
    """
    counter = collections.Counter(read_text_words(filename))
    ordered_pairs = sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if len(ordered_pairs) > limit:
        ordered_pairs = ordered_pairs[:limit]

    words = [word for word, _ in ordered_pairs]

    return dict(zip(words, range(len(words))))


def read_text_words(filename):
    """ Read text file and iterate through all words.

    Parameters
    ----------
    filename : str
        Filename of the input data

    Yields
    ------
    str
        The next word in a sentence.
    """
    for line in read_text_sents(filename):
        for word in line.split():
            yield word


def read_text_sents(filename):
    """ Read text file and iterate through all sentences.

    Parameters
    ----------
    filename : str
        Filename of the input data

    Yields
    ------
    str
        The next sentence in the document.
    """
    with open(filename, "r") as fin:
        for line in fin.readlines():
            yield line


def tokenize(sent, vocab):
    """ Convert a sentence to a list of tokens.

    Note
    ----
    The tokes are represented by the index of the words in a given vocabulary. Words that can't
    be found in the dictionary are represented by the largest index + 1.


    Parameters
    ----------
    sent : list of str
        Input sentence.

    vocab : dict
        Dictionary of the vocabulary.

    Returns
    -------
    list of list
        List of tokenized sentence (represented by a list).
    """
    unk_index = len(vocab.keys())
    tokens = [vocab.get(word, unk_index) for word in sent.split()]

    return tokens


def slide_window(arr, n):
    """ Return an iterator of size n sliding window.

    Parameters
    ----------
    arr : list
        Input list
    n : int
        Size of the window.

    Yields
    ------
    list
        Sliding window of size n
    """
    for i in range(len(arr)-n+1):
        yield arr[i:i+n]


def split_toksents(toksents):
    """ Split the tokens within a widow into (x, y) pairs.

    Note
    ----
        Formula:

        e_t => y
        e^{t-1}_{t-n+1} => x
        x = [\phi(e_{t-n+1});\phi(e_{t-n+2});...;\phi(e_{t-2});\phi(e_{t-1})]

        This method does not convert (x, y) into one-hot encodings. It only keeps the index of
        non-zero elements.

    Parameters
    ----------
    toksents : list of list or 2D numpy array of shape (n_samples, n_grams)
        The whole tokenized dataset.
    Returns
    -------
    x : 2D numpy array of shape (n_samples, n_grams - 1)
        x features
    y : 1D numpy array of shape (n_samples,)
        y outcomes
    """
    arr = np.array(toksents)
    x = arr[:, :-1]
    y = arr[:, -1]

    return x, y
