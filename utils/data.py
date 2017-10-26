""" Common Data Loading methods for Languag Models.
"""
import collections
import sys

import numpy as np


def build_vocab(filename, limit=None):
    """ Build a vocabulary from the corpus.

    Parameters
    ----------
    filename : str
        Filename of the input data
    limit : int, optional
        Limit the number of words in the vocabulary. Default value is None (no limit)

    Returns
    -------
    dict
        Python dictionary of {<word>: <index>} pairs.
    """
    if limit is None:
        limit = sys.maxsize

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
    toksents : list of list or 2D numpy array of shape (n_samples, n_grams + 1)
        The whole tokenized dataset.
    Returns
    -------
    x : numpy array of shape (n_samples, n_grams)
        Features
    y : numpy array of shape (n_samples,)
        Outcomes
    """
    arr = np.array(toksents)
    x = arr[:, :-1]
    y = arr[:, -1]

    return x, y


def get_embedding(vocab):
    """ Read Pre-trained fasttext embedding.

    Parameters
    ----------
    vocab : dict
        Dictionary of the vocabulary.

    Returns
    -------
    numpy array of shape (n_words, embedding_dim)
        vector embeddings
    """
    prefix = "data/fasttext/"
    embedding_filename = prefix + "wiki-news-300d-1M.vec"

    all_embedding = {}
    with open(embedding_filename, "r") as fin:
        fin.readline()
        for line in fin.readlines():
            line_data = line.split()
            token, vec = line_data[0], line_data[1:]
            all_embedding[token] = np.array(vec).astype(np.float32)

    n_words = len(vocab.keys())
    embedding_dim = 300
    embedding = np.zeros((n_words, embedding_dim))
    for word, token in vocab.items():
        embedding[token] = all_embedding.get(
            word, np.random.randn(embedding_dim))

    return embedding


def prepare_data(n_grams, max_num_sents=None, max_num_voacb=None):
    """ Prepare trainig & testing data for language models.

    Parameters
    ----------
    n_grams : int
        Number of previous words (n-grams) in a sliding window.
    max_num_sents : int, optional
        Maximum number of sentences to read. Default to None (no limit).
    max_num_voacb : int, optional
        Maximum number of words to use in vocabulary. Default to None (no limit).

    Returns
    -------
    vocab : dict
        Python dictionary of {<word>: <index>} pairs.
    x : numpy array of shape (n_train_samples, n_grams)
        Training Features
    y : numpy array of shape (n_train_samples,)
        Trainig Labels
    x_test : numpy array of shape (n_test_samples, n_grams)
        Testing Features
    y_test : numpy array of shape (n_test_samples,)
        Testing Labels
    """
    prefix = "data/iwslt/en-de/"
    train_file = prefix + "train.en-de.tok.filt.en"
    test_file = prefix + "test.en-de.tok.en"

    vocab = build_vocab(train_file, max_num_voacb)

    train_toksents = [tokenize(sent, vocab) for sent in read_text_sents(train_file)]
    test_toksents = [tokenize(sent, vocab) for sent in read_text_sents(test_file)]

    if max_num_sents is None:
        train_max_num_sents = len(train_toksents) + 1
        test_max_num_sents = len(test_toksents) + 1
    else:
        train_max_num_sents = test_max_num_sents = max_num_sents

    train_data = []
    for sent in train_toksents[:train_max_num_sents]:
        train_data.extend(slide_window(sent, n_grams+1))
    test_data = []
    for sent in test_toksents[:test_max_num_sents]:
        test_data.extend(slide_window(sent, n_grams+1))

    x, y = split_toksents(train_data)
    print("Total number of training samples: ", x.shape[0])
    x_test, y_test = split_toksents(test_data)
    print("Total number of testing samples: ", x_test.shape[0])

    return vocab, x, y, x_test, y_test
