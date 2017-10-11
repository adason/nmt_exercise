""" Feed Forward Language Model.
"""
from sklearn.metrics import accuracy_score

from utils.data import prepare_data


class FeedForwardModel:
    """ Feed Forward Neural Network Language Model

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
    """
    def __init__(self):
        if random_state:
            np.random.seed(random_state)

        self.n_grams = n_grams
        self.n_vocab = n_vocab


def main():
    """ Main
    """
    # Some limiting Parameters
    n_grams = 2
    max_num_sents = 1000
    max_num_voacb = 4000

    vocab, x, y, x_test, y_test = prepare_data(n_grams, max_num_sents, max_num_voacb)
    n_vocab = len(vocab.keys()) + 1 # The extra space reserved for unknown word.

    model = FeedForwardModel(n_grams, n_vocab)
    # model.fit(x, y, 5)
    #
    # y_pred = model.predict(x_test)
    # print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
