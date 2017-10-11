""" Implement Log-linear Language Model.
"""
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.data import prepare_data
from utils.trans import binarize, softmax


def neg_log_loss(y, p, eps=1e-15):
    """ Compute the negative log loss (cross entropy loss).

    Note
    ----
        Formula: loss = \sum_{i=0}^{N-1}\sum_{k=0}^{K-1} y_ik * log (p_ik)

    Parameters
    ----------
    y : numpy array of shape (n_samples,)
        Multi-class labels (not one-hot encodings)
    p : numpy array of shape (n_samples, n_labels)
        Softmax transformed predictions.

    Returns
    -------
    np.float64
        Total sum of negative log losses for all samples.

    """
    n_vocab = p.shape[1]
    y = binarize(y, n_vocab)
    # Log loss is undefined for p=0 or p=1,
    # so probabilities are clipped to (eps, 1-eps).
    p = np.clip(p, eps, 1 - eps)

    return -(y * np.log(p)).sum()


class LogLinearModel:
    """ Log Linear Model.

    Parameters
    ----------
    n_grams : int
        Number of previous words in this language model.
    n_vocab : int
        Number of words in the whole vocabulary.
    random_state : int, optional
        Integer used to assign initial random seed.

    Attributes
    ----------
    n_grams : int
        Number of previous words in this language model.
    n_vocab : int
        Number of words in the whole vocabulary.
    w : numpy array of shape (n_grams, n_vocab, n_vocab)
        The weights for this n_grams language model, initialized randomly.
    b : numpy array of shape (n_vocab,)
        The biases term, initialized randomly.
    """
    def __init__(self, n_grams, n_vocab, random_state=None):
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

        Parameters
        ----------
        x : numpy array of shape (n_samples, n_grams)
            Features
        y : numpy array of shape (n_samples,)
            Outcomes
        epochs : int
            Maximum number of epochc for the SGD algorithm before convergence.
        batch_size : int
            Number of samples in each mini-batch SGD.

        Returns
        -------
        list
            History of loss computed at the end of each epoch.
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

        Parameters
        ----------
        x : numpy array of shape (n_samples, n_grams)
            Features

        Returns
        -------
        numpy array of shape (n_samples,)
            The predicted index for each sample using the largest probability.
        """
        p = self.predict_proba(x)

        return np.argmax(p, axis=1)

    def predict_proba(self, x):
        """ Predict the probability.

        Parameters
        ----------
        x : numpy array of shape (n_samples, n_grams)
            Features

        Returns
        -------
        numpy array of shape (n_samples, n_vocab)
            Softmax transformed probability predictions.
        """
        s = self._linear_trans(x, self.w, self.b)
        p = softmax(s, axis=1)

        return p

    def _fit_one_epoch(self, x, y, batch_size, eta=0.1):
        """ Fit one epoch using minibatch SGD.

        Note
        ----
            This method implemented minibatch SGD without momentum. It shuffles the training sample
            to create different minibatches at different epoch.

        Parameters
        ----------
        x : numpy array of shape (n_samples, n_grams)
            Features
        y : numpy array of shape (n_samples,)
            Outcomes
        batch_size : int
            Number of samples in each mini-batch SGD.
        eta : float
            Learning rate.

        Returns
        -------
        np.float64
            Total loss after one epoch training.
        """
        y_onehot = binarize(y, self.n_vocab)

        losses = []
        shuffled_idx = np.random.permutation(x.shape[0])
        split_idx = [batch_size*(i+1) for i in range(int(x.shape[0] / batch_size))]
        for batch_idx in tqdm(np.split(shuffled_idx, split_idx)):
            x_bat = np.take(x, batch_idx, axis=0)
            y_bat = np.take(y, batch_idx, axis=0)
            y_onehot_bat = np.take(y_onehot, batch_idx, axis=0)
            p = softmax(self._linear_trans(x_bat, self.w, self.b), axis=1)
            p_m_onehot_et = p - y_onehot_bat
            self.b -= eta * np.mean(p_m_onehot_et, axis=0)
            self.w -= eta * self._grad_w(x_bat, p_m_onehot_et)

            loss_bat = self._total_loss(x_bat, y_bat, self.w, self.b)
            losses.append(loss_bat)

        return np.sum(losses)

    @staticmethod
    def _grad_w(x_bat, p_m_onehot_et):
        """ Compute the gradient of weithgs w within a minibatch.

        Note
        ----
            Formula: d(loss)/d(w_j) = x_j(p - onehot(e_t))

        Parameters
        ----------
        x_bat : numpy array of shape (n_samples, n_grams)
            Featrues within a minibatch
        p_m_onehot_et : numpy array of shape (n_samples, n_vocab)
            Numerical value of P - onthot(e_t)

        Returns
        -------
        numpy array of shape (n_grams, n_vocab, n_vocab)
            Minibatch averaged gradient of weights w.
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
    def _linear_trans(x, w, b):
        """ Compute the linear transformation.

        Note
        ----
            Formula: s_j = \sum_{j; x_j != 0} w_{.,j} * x_j + b_j

        Parameters
        ----------
        x : numpy array of shape (n_samples, n_grams)
            Features
        w : numpy array of shape (n_grams, n_vocab, n_vocab)
            Weights
        b : numpy array of shape (n_vocab,)
            Biases

        Returns
        -------
        numpy array of shape (n_samples, n_vocab)
            The linear transformation of wx+b.
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
    def _total_loss(x, y, w, b):
        """ Compute the total loss.

        Parameters
        ----------
        x : numpy array of shape (n_samples, n_grams)
            Features
        y : numpy array of shape (n_samples,)
            Outcomes
        w : numpy array of shape (n_grams, n_vocab, n_vocab)
            Weights
        b : numpy array of shape (n_vocab,)
            Biases

        Returns
        -------
        np.float64
            Total negative log loss.
        """
        s = LogLinearModel._linear_trans(x, w, b)
        p = softmax(s, axis=1)

        return neg_log_loss(y, p)


def main():
    """ Main
    """
    # Some limiting Parameters
    n_grams = 2
    max_num_sents = 1000
    max_num_voacb = 4000

    vocab, x, y, x_test, y_test = prepare_data(n_grams, max_num_sents, max_num_voacb)
    n_vocab = len(vocab.keys()) + 1 # The extra space reserved for unknown word.

    model = LogLinearModel(n_grams, n_vocab)
    model.fit(x, y, 5)

    y_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
