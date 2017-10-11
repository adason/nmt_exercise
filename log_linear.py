""" Implement Log-linear Model in Chapter 4.
"""
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.data import build_vocab, tokenize, read_text_sents, slide_window, split_toksents
from utils.trans import binarize, softmax


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
    # Some limiting Parameters
    n_grams = 2
    max_num_sents = 1000
    max_num_voacb = 4000

    prefix = "data/iwslt/en-de/"
    train_file = prefix + "train.en-de.tok.filt.en"
    test_file = prefix + "test.en-de.tok.en"

    vocab = build_vocab(train_file, max_num_voacb)
    n_vocab = len(vocab.keys()) + 1 # The extra space reserved for unknown word.
    train_toksents = [tokenize(sent, vocab) for sent in read_text_sents(train_file)]
    test_toksents = [tokenize(sent, vocab) for sent in read_text_sents(test_file)]

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
