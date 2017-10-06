""" Implement Log-linear Model in Chapter 4.
"""
import collections

import numpy as np


def build_vocab(data_file):
    """ Build a vocabulary from the corpus.
    """
    counter = collections.Counter(read_text_words(data_file))
    ordered_pairs = sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True)
    words = [word for word, _ in ordered_pairs]

    return dict(zip(words, range(len(words))))


def read_text_words(data_file):
    """ Read text file and iterate through all words.
    """
    for line in read_text_sents(data_file):
        for word in line.split():
            yield word


def read_text_sents(data_file):
    """ Read text file and iterate through all sentences.
    """
    with open(data_file, "r") as fin:
        for line in fin.readlines():
            yield line


def tokenize(sent, vocab):
    """ Convert a sentence to a list of ids.
    """
    unk_index = len(vocab.keys())
    tokens = [vocab.get(word, unk_index) for word in sent.split()]

    return tokens


def slide_window(arr, n):
    """ Return an iterator of size n sliding window.
    """
    for i in range(len(arr)-n+1):
        yield arr[i:i+n]


# def binarize_toksents(toksents, n_vocab):
#     """ Split the tokens within a widow into (x, y) pairs, encoded in sparse matrix.
#         e_t => y
#         e^{t-1}_{t-n+1} => x
#         x = [\phi(e_{t-n+1});\phi(e_{t-n+2});...;\phi(e_{t-2});\phi(e_{t-1})]
#     """
#     x_indptr, y_indptr = [0], [0]
#     x_indices, y_indices = [], []
#     x_data, y_data = [], []
#     for sent in toksents:
#         # x
#         x_offset = 0
#         for tok in sent[:-1]:
#             x_indices.append(x_offset + tok)
#             x_data.append(1)
#             x_offset += n_vocab
#         x_indptr.append(len(x_indices))
#         # y
#         tok = sent[-1]
#         y_indices.append(tok)
#         y_data.append(1)
#         y_indptr.append(len(y_indices))
#
#     n_rows = len(toksents)
#     n_x_cols = (len(toksents[0]) - 1) * n_vocab
#     x = csr_matrix((x_data, x_indices, x_indptr), shape=(n_rows, n_x_cols), dtype=int)
#
#     n_y_cols = n_vocab
#     y = csr_matrix((y_data, y_indices, y_indptr), shape=(n_rows, n_y_cols), dtype=int)
#
#     return x, y


def split_toksents(toksents):
    """ Split the tokens within a widow into (x, y) pairs.
        e_t => y
        e^{t-1}_{t-n+1} => x
        x = [\phi(e_{t-n+1});\phi(e_{t-n+2});...;\phi(e_{t-2});\phi(e_{t-1})]

        Note that this method does not convert (x, y) into one-hot encodings. It only keeps
        the index of non-zero elements.
    """
    arr = np.array(toksents)
    x = arr[:, :-1]
    y = arr[:, -1]

    return x, y


def linear_trans(x, w, b):
    """ Compute the linear transformation.
        x: token indicies; numpy array of shape (n_samples, n_grams)
        w: weight matrix; numpy array of shape (n_grams, n_vocab, n_vocab)
        b: bias; numpy array of shape (n_vocab,)
        Formula: s_j = \sum_{j; x_j != 0} w_{.,j} * x_j + b_j
    """
    b = np.expand_dims(b, axis=0)
    wx = []
    for i in range(x.shape[0]):
        x_ = x[i, :]  # (n_grams,)
        w_ = [w[idx, :, j] for idx, j in enumerate(x_)]  # [(n_vocab,), (n_vocab,), ..]
        wx_ = np.vstack(w_).sum(axis=0) # (n_vocab, )
        wx.append(wx_)

    wx = np.vstack(wx)

    return wx + b


def binarize(y):
    """ Convert multi-class y labels into one-hot encoded vectors.
    """
    y = np.array(y)
    n_cols = np.unique(y).shape[0]
    n_rows = y.shape[0]
    bin_matrix = np.zeros((n_rows, n_cols))
    for i, idx in enumerate(y):
        bin_matrix[i, idx] = 1

    return bin_matrix


def log_loss(y, p, eps=1e-15):
    """ Compute the log loss (cross entropy loss).

        y: multi-class labels (not one-hot encodings)
        p: softmax transformed predictions.
        N: number of samples
        K: number of labels
        loss = \sum_{i=0}^{N-1}\sum_{k=0}^{K-1} y_ik * log (p_ik)
    """
    y = binarize(y)
    # Log loss is undefined for p=0 or p=1,
    # so probabilities are clipped to (eps, 1-eps).
    p = np.clip(p, eps, 1 - eps)

    return -(y * np.log(p)).sum()


def softmax(s, axis=None):
    """ Compute softmax transformation along the axis.

        p_i = exp(s_i - max(s)) / \sum_i exp(s_i - max(s))
    """
    s = np.atleast_2d(s)

    if axis is None:
        axis = s.ndim - 1
    s = s - np.expand_dims(np.max(s, axis), axis)
    s = np.exp(s)
    s_sum = np.expand_dims(np.sum(s, axis), axis)
    p = s / s_sum

    return p.squeeze()


def total_loss(x, y, w, b):
    s = linear_trans(x, w, b)
    p = softmax(s, axis=1)

    return log_loss(y, p)


def main():
    prefix = "data/iwslt/en-de/"
    train_file = prefix + "train.en-de.tok.filt.en"
    test_file = prefix + "test.en-de.tok.en"

    vocab = build_vocab(train_file)
    n_vocab = len(vocab.keys()) + 1 # The extra space reserved for unknown word.
    train_toksents = [tokenize(sent, vocab) for sent in read_text_sents(train_file)]
    test_toksents = [tokenize(sent, vocab) for sent in read_text_sents(test_file)]

    # print(train_toksents[:5])
    # print(test_toksents[:5])

    # print(train_toksents[0])
    # print(list(slide_window(train_toksents[0], 3)))

    # x, y = binarize_toksents([[0, 1, 2],[2, 1, 0]], 3)
    # print(x.toarray())
    # print(y.toarray())

    x, y = split_toksents([[0, 1, 2],[2, 1, 0]])
    print(x, y)
    print(softmax([1,2,3]))

    w = np.array([
        [[0.2, 0.6, 0.3],
         [0.5, 0.1, 0.4],
         [0.3, 0.3, 0.3]],
        [[0.1, 0.6, 0.4],
         [0.6, 0.1, 0.4],
         [0.3, 0.3, 0.2]]
    ])
    b = np.array([0.1, 0.1, 0.1])
    print(linear_trans(x, w, b))


    # y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # p = np.array([[0.9, 0.1, 0.2], [0.3, 0.6, 0.2], [0.5, 0.4, 0.8]])
    # print(log_loss(y, p))


if __name__ == "__main__":
    main()
