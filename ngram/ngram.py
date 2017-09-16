""" Modified Kneser Ney
"""
import collections

import numpy as np
import nltk


class KneserNeyModel:
    """ The Modified Kneser Ney Smoothing
    """
    def __init__(self, highest_order=2):
        assert highest_order >= 2
        self.highest_order = highest_order

        self.words = []
        # Store the probablitiy by: probs[("x", "y")] = p(y|x)
        self.probs = collections.defaultdict(float)

    def train(self, sents):
        all_ngram_counts = self._get_all_ngram_counts(sents)
        self.words = [word[0] for word in all_ngram_counts[0].keys()]
        self._calc_unigram_probs(all_ngram_counts)
        self._calc_probs(all_ngram_counts)

    def predict_next_word(self, sentence):
        context = self._get_context(sentence)
        word_probs = []
        for word in self.words:
            key = tuple(context + [word])
            word_probs.append(self.probs[key])

        return self.words[np.argmax(word_probs)]

    def _get_context(self, sentence):
        return sentence[(len(sentence) - self.highest_order + 1):]

    def _get_all_ngram_counts(self, sents):
        all_ngram_counts = []
        for n in range(1, self.highest_order + 1):
            ngrams = [
                ngram for sent in sents for ngram in nltk.ngrams(sent, n)
            ]
            all_ngram_counts.append(collections.Counter(ngrams))

        return all_ngram_counts

    def _get_discount(self, c, hist_of_ngram_counts):
        n1 = hist_of_ngram_counts[1]
        n2 = hist_of_ngram_counts[2]
        n3 = hist_of_ngram_counts[3]
        n4 = hist_of_ngram_counts[4]
        y = 1.0 * n1 / (n1 + 2 * n2)

        if c == 0:
            return 0
        elif c == 1:
            return 1.0 - 2.0 * y * n2 / n1
        elif c == 2:
            return 2.0 - 3.0 * y * n3 / n2
        elif c >= 3:
            return 3.0 - 4.0 * y * n4 / n3

    def _calc_unigram_probs(self, all_ngram_counts):
        bigram_counts = all_ngram_counts[1]

        n_dotwi = collections.defaultdict(int)
        for w, c in bigram_counts.items():
            if c > 0:
                n_dotwi[w[1:]] += 1
        n_dotdot = sum(n_dotwi.values())

        for wi, val in n_dotwi.items():
            pkn_wi = 1.0 * val / n_dotdot
            self.probs[wi] = pkn_wi

        return self.probs

    def _calc_probs(self, all_ngram_counts):
        n1_wdot = collections.defaultdict(int)
        n2_wdot = collections.defaultdict(int)
        n3_wdot = collections.defaultdict(int)
        for order in range(2, self.highest_order + 1):
            thisgram_counts = all_ngram_counts[order - 1]
            prevgram_counts = all_ngram_counts[order - 2]
            hist_of_ngram_counts = collections.Counter(
                val for val in thisgram_counts.values() if val <= 4)

            for prefix in prevgram_counts.keys():
                for w in self.words:
                    key = tuple(list(prefix) + [w])
                    if thisgram_counts[key] == 1:
                        n1_wdot[prefix] += 1
                    elif thisgram_counts[key] == 2:
                        n2_wdot[prefix] += 1
                    elif thisgram_counts[key] >= 3:
                        n3_wdot[prefix] += 1

            for prefix in prevgram_counts.keys():
                for w in self.words:
                    key = tuple(list(prefix) + [w])
                    suffix = key[1:]

                    discount = self._get_discount(
                        thisgram_counts[key], hist_of_ngram_counts)
                    d1 = self._get_discount(1, hist_of_ngram_counts)
                    d2 = self._get_discount(2, hist_of_ngram_counts)
                    d3 = self._get_discount(3, hist_of_ngram_counts)
                    gamma = 1.0 * (
                        d1 * n1_wdot[prefix] +
                        d2 * n2_wdot[prefix] +
                        d3 * n3_wdot[prefix]
                    ) / prevgram_counts[prefix]

                    discounted_prob = (
                        1.0 * (thisgram_counts[key] - discount) /
                        prevgram_counts[prefix]
                    )
                    pkn = discounted_prob + gamma * self.probs[suffix]
                    self.probs[key] = pkn
