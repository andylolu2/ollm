# mypy: ignore-errors
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract class module which defines the interface for our HypernymySuite model.
"""

import numpy as np
import scipy.sparse as sparse


class HypernymySuiteModel(object):
    """
    Base class for all hypernymy suite models.

    To use this, must implement these methods:

        predict(self, hypo: str, hyper: str): float, which makes a
            prediction about two words.
        vocab: dict[str, int], which tells if a word is in the
            vocabulary.

    Your predict method *must* be prepared to handle OOV terms, but it may
    returning any sentinel value you wish.

    You can optionally implement
        predict_many(hypo: list[str], hyper: list[str]: array[float]

    The skeleton method here will just call predict() in a for loop, but
    some methods can be vectorized for improved performance. This is the
    actual method called by the evaluation script.
    """

    vocab = {}

    def __init__(self):
        raise NotImplementedError

    def predict(self, hypo, hyper):
        """
        Core modeling procedure, estimating the degree to which hypo is_a hyper.

        This is an abstract method, describing the interface.

        Args:
            hypo: str. A hypothesized hyponym.
            hyper: str. A hypothesized hypernym.

        Returns:
            float. The score estimating the degree to which hypo is_a hyper.
                Higher values indicate a stronger degree.
        """
        raise NotImplementedError

    def predict_many(self, hypos, hypers):
        """
        Make predictions for many pairs at the same time. The default
        implementation just calls predict() many times, but many models
        benefit from vectorization.

        Args:
            hypos: list[str]. A list of hypothesized hyponyms.
            hypers: list[str]. A list of corresponding hypothesized hypernyms.
        """
        result = []
        for x, y in zip(hypos, hypers):
            result.append(self.predict(x, y))
        return np.array(result, dtype=np.float32)


class Precomputed(HypernymySuiteModel):
    """
    A model which uses precomputed prediction, read from a TSV file.
    """

    def __init__(self, precomputed):
        self.vocab = {"<OOV>": 0}
        self.lookup = {}
        with open(precomputed) as f:
            for line in f:
                w1, w2, sim, is_oov = line.strip().split("\t")
                if w1 == "hypo" and w2 == "hyper":
                    # header, ignore it
                    continue
                if is_oov == "1" or is_oov.lower() in ("t", "true"):
                    # Don't read in oov predictions
                    continue
                if w1 not in self.vocab:
                    self.vocab[w1] = len(self.vocab)
                if w2 not in self.vocab:
                    self.vocab[w2] = len(self.vocab)
                sim = float(sim)
                self.lookup[(self.vocab[w1], self.vocab[w2])] = sim

    def predict(self, hypo, hyper):
        x = self.vocab.get(hypo, 0)
        y = self.vocab.get(hyper, 0)
        return self.lookup.get((x, y), 0.0)


class PatternBasedModel(HypernymySuiteModel):
    """
    Basis class for all Hearst-pattern based approaches.
    """

    def __init__(self, csr_m, vocab):
        self.vocab = vocab
        self.p_w = csr_m.sum(axis=1).A[:, 0]
        self.p_c = csr_m.sum(axis=0).A[0, :]
        self.matrix = csr_m.todok()
        self.N = self.p_w.sum()

    def predict(self, hypo, hyper):
        raise NotImplementedError("Abstract class")

    def __str__(self):
        raise NotImplementedError("Abstract class")


class RawCountModel(PatternBasedModel):
    """
    P(x, y) model which uses raw counts.
    """

    def predict(self, hypo, hyper):
        L = self.vocab.get(hypo, 0)
        R = self.vocab.get(hyper, 0)
        return self.matrix[(L, R)]

    def __str__(self):
        return "raw"


class PPMIModel(RawCountModel):
    """
    PPMI(x, y) model which uses a PPMI-transformed Hearst patterns for
    predictions.
    """

    def __init__(self, csr_m, vocab):
        # first read in the normal stuff
        super().__init__(csr_m, vocab)
        # now let's transform the matrix
        tr_matrix = sparse.dok_matrix(self.matrix.shape)
        # actually do the transformation
        for l, r in self.matrix.keys():
            pmi_lr = (
                np.log(self.N)
                + np.log(self.matrix[(l, r)])
                - np.log(self.p_w[l])
                - np.log(self.p_c[r])
            )
            # ensure it's /positive/ pmi
            ppmi_lr = np.clip(pmi_lr, 0.0, 1e12)
            tr_matrix[(l, r)] = ppmi_lr
        self.matrix = tr_matrix

    def __str__(self):
        return "ppmi"


class _SvdMixIn(object):
    """
    Abstract mixin, do not use directly. Computes the SVD on top of the matrix
    from the superclass (may only be mixed in with a PatternBasedModel).
    """

    def __init__(self, csr_m, vocab, k: int):
        # First make sure the matrix is loaded
        super(_SvdMixIn, self).__init__(csr_m, vocab)
        self.k = k
        U, S, V = sparse.linalg.svds(self.matrix.tocsr(), k=k)
        self.U = U.dot(np.diag(S))
        self.V = V.T

    def predict(self, hypo, hyper):
        L = self.vocab.get(hypo, 0)
        R = self.vocab.get(hyper, 0)
        return self.U[L].dot(self.V[R])

    def predict_many(self, hypos, hypers):
        lhs = [self.vocab.get(x, 0) for x in hypos]
        rhs = [self.vocab.get(x, 0) for x in hypers]

        retval = np.sum(self.U[lhs] * self.V[rhs], axis=1)
        return retval

    def __str__(self):
        return "svd" + super(_SvdMixIn, self).__str__()


class SvdRawModel(_SvdMixIn, RawCountModel):
    """
    sp(x,y) model presented in the paper. This is an svd over the raw counts.
    """

    pass


class SvdPpmiModel(_SvdMixIn, PPMIModel):
    """
    spmi(x,y) model presented in the paper. This is the svd over the ppmi matrix.
    """

    pass
