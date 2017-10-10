""" Implement Log-linear Model in Chapter 4.
"""
import collections

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


MAX_NUM_VOCAB = 4000


def build_vocab(data_file):
    """ Build a vocabulary from the corpus.
    """
    counter = collections.Counter(read_text_words(data_file))
    ordered_pairs = sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if len(ordered_pairs) > MAX_NUM_VOCAB:
        ordered_pairs = ordered_pairs[:MAX_NUM_VOCAB]

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


def binarize(y, n_vocab):
    """ Convert multi-class y labels into one-hot encoded vectors.
    """
    y = np.array(y)
    n_samples = y.shape[0]
    bin_matrix = np.zeros((n_samples, n_vocab))
    for i, idx in enumerate(y):
        bin_matrix[i, idx] = 1

    return bin_matrix


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


def log_loss(y, p, eps=1e-15):
    """ Compute the log loss (cross entropy loss).

        y: multi-class labels (not one-hot encodings)
        p: softmax transformed predictions.
        N: number of samples
        K: number of labels
        loss = \sum_{i=0}^{N-1}\sum_{k=0}^{K-1} y_ik * log (p_ik)
    """
    n_vocab = p.shape[1]
    y = binarize(y, n_vocab)
    # Log loss is undefined for p=0 or p=1,
    # so probabilities are clipped to (eps, 1-eps).
    p = np.clip(p, eps, 1 - eps)

    return -(y * np.log(p)).sum()


class LogLinearModel:
    """ Log Linear Model.
    """
    def __init__(self, n_grams, n_vocab, random_state=None):
        """ Initialize parameters.

            :n_grams:
            :n_vocab:
            :random_state:
        """
        if random_state:
            np.random.seed(random_state)

        self.n_grams = n_grams
        self.n_vocab = n_vocab

        self.w = np.random.random_sample((n_grams, n_vocab, n_vocab))
        self.b = np.random.random_sample((n_vocab,))

    def __repr__(self):
        return "w: {}\n b: {}".format(self.w, self.b)

    def fit(self, x, y, epochs=10, batch_size=32):
        """ Fit the parameters.
        """
        loss_history = [np.Inf]
        for e in range(epochs):
            print("Training epoch {}".format(e))
            loss = self._fit_one_epoch(x, y, batch_size)
            loss_history.append(loss)
            print("Done training epoch: {}. Loss: {}".format(e, loss))

            if abs(loss_history[-2] - loss_history[-1]) < 1e-6:
                break

        return loss_history

    def predict(self, x):
        """ Predict using the highest probability.
        """
        p = self.predict_proba(x)

        return np.argmax(p, axis=1)

    def predict_proba(self, x):
        """ Predict the probability.
        """
        s = self.linear_trans(x, self.w, self.b)
        p = softmax(s, axis=1)

        return p

    def _fit_one_epoch(self, x, y, batch_size, eta=0.1):
        """ Fit one epoch.
        """
        y_onehot = binarize(y, self.n_vocab)

        losses = []
        shuffled_idx = np.random.permutation(x.shape[0])
        split_idx = [batch_size*(i+1) for i in range(int(x.shape[0] / batch_size))]
        for batch_idx in tqdm(np.split(shuffled_idx, split_idx)):
            x_bat = np.take(x, batch_idx, axis=0)
            y_bat = np.take(y, batch_idx, axis=0)
            y_onehot_bat = np.take(y_onehot, batch_idx, axis=0)
            p = softmax(self.linear_trans(x_bat, self.w, self.b), axis=1)
            p_m_onehot_et = p - y_onehot_bat
            self.b -= eta * np.mean(p_m_onehot_et, axis=0)
            self.w -= eta * self._grad_w(x_bat, p_m_onehot_et)

            loss_bat = self.total_loss(x_bat, y_bat, self.w, self.b)
            losses.append(loss_bat)

        return np.sum(losses)

    @staticmethod
    def _grad_w(x_bat, p_m_onehot_et):
        """ Compute the gradient of w within a minibatch.

            x_bat: shape (n_samples, n_grams)
            p_m_onehot_et: shape (n_samples, n_vocab)

            dloss/dw_j = x_j(p - onehot(e_t))
        """
        n_samples = x_bat.shape[0]
        n_grams = x_bat.shape[1]
        n_vocab = p_m_onehot_et.shape[1]
        gw = np.zeros((n_samples, n_grams, n_vocab, n_vocab))
        for i in range(x_bat.shape[0]):
            x_bat_ = x_bat[i, :]  # (n_grams,)
            for idx, j in enumerate(x_bat_):
                gw[i, idx, :, j] = p_m_onehot_et[i, j]

        return np.mean(gw, axis=0)


    @staticmethod
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

    @staticmethod
    def total_loss(x, y, w, b):
        """ Compute the total loss.
        """
        s = LogLinearModel.linear_trans(x, w, b)
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

    # Some limiting Parameters
    n_grams = 2
    max_num_sents = 1000
    train_data = []
    for sent in train_toksents[:max_num_sents]:
        train_data.extend(slide_window(sent, n_grams+1))

    x, y = split_toksents(train_data)
    print("Total number of training samples: ", x.shape[0])

    model = LogLinearModel(n_grams, n_vocab)
    model.fit(x, y, 5)

    test_data = []
    for sent in test_toksents[:max_num_sents]:
        test_data.extend(slide_window(sent, n_grams+1))

    x_test, y_test = split_toksents(test_data)
    y_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
