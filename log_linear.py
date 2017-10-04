""" Implement Log-linear Model in Chapter 4.
"""
import collections
from scipy.sparse import csr_matrix


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


def binarize_toksents(toksents, n_vocab):
    """ Split the tokens within a widow into (x, y) pairs, encoded in sparse matrix.
        e_t => y
        e^{t-1}_{t-n+1} => x
        x = [\phi(e_{t-n+1});\phi(e_{t-n+2});...;\phi(e_{t-2});\phi(e_{t-1})]
    """
    x_indptr, y_indptr = [0], [0]
    x_indices, y_indices = [], []
    x_data, y_data = [], []
    for sent in toksents:
        # x
        x_offset = 0
        for tok in sent[:-1]:
            x_indices.append(x_offset + tok)
            x_data.append(1)
            x_offset += n_vocab
        x_indptr.append(len(x_indices))
        # y
        tok = sent[-1]
        y_indices.append(tok)
        y_data.append(1)
        y_indptr.append(len(y_indices))

    n_rows = len(toksents)
    n_x_cols = (len(toksents[0]) - 1) * n_vocab
    x = csr_matrix((x_data, x_indices, x_indptr), shape=(n_rows, n_x_cols), dtype=int)

    n_y_cols = n_vocab
    y = csr_matrix((y_data, y_indices, y_indptr), shape=(n_rows, n_y_cols), dtype=int)

    return x, y


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

    x, y = binarize_toksents([[0,1,2],[2,1,0]], 3)
    print(x.toarray())
    print(y.toarray())


if __name__ == "__main__":
    main()
