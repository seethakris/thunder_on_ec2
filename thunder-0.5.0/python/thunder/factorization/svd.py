"""
Class for performing Singular Value Decomposition
"""

from numpy import zeros, shape

from thunder.utils.common import checkParams
from thunder.rdds.series import Series
from thunder.rdds.matrices import RowMatrix


class SVD(object):
    """
    Singular value decomposiiton on a distributed matrix.

    Parameters
    ----------
    k : int, optional, default = 3
        Number of singular vectors to estimate

    method : string, optional, default = "auto"
        Whether to use a direct or iterative method. If set to 'auto',
        will select preferred method based on dimensionality.

    maxIter : int, optional, default = 20
        Maximum number of iterations if using an iterative method

    tol : float, optional, default = 0.00001
        Tolerance for convergence of iterative algorithm

    Attributes
    ----------
    `u` : RowMatrix, nrows, each of shape (k,)
        Left singular vectors

    `s` : array, shape(nrows,)
        Singular values

    `v` : array, shape (k, ncols)
        Right singular vectors
    """
    def __init__(self, k=3, method="auto", maxIter=20, tol=0.00001):
        self.k = k
        self.method = method
        self.maxIter = maxIter
        self.tol = tol
        self.u = None
        self.s = None
        self.v = None

    def calc(self, mat):
        """
        Calcuate singular vectors

        Parameters
        ----------
        mat :  Series or a subclass (e.g. RowMatrix)
            Matrix to compute singular vectors from

        Returns
        ----------
        self : returns an instance of self.
        """

        from numpy import argsort, dot, outer, random, sqrt, sum
        from scipy.linalg import inv, orth
        from numpy.linalg import eigh

        if not (isinstance(mat, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if not (isinstance(mat, RowMatrix)):
            mat = mat.toRowMatrix()

        checkParams(self.method, ['auto', 'direct', 'em'])

        if self.method == 'auto':
            if len(mat.index) < 750:
                method = 'direct'
            else:
                method = 'em'
        else:
            method = self.method

        if method == 'direct':

            # get the normalized gramian matrix
            cov = mat.gramian() / mat.nrows

            # do a local eigendecomposition
            eigw, eigv = eigh(cov)
            inds = argsort(eigw)[::-1]
            s = sqrt(eigw[inds[0:self.k]]) * sqrt(mat.nrows)
            v = eigv[:, inds[0:self.k]].T

            # project back into data, normalize by singular values
            u = mat.times(v.T / s)

            self.u = u
            self.s = s
            self.v = v

        if method == 'em':

            # initialize random matrix
            c = random.rand(self.k, mat.ncols)
            niter = 0
            error = 100

            # define an accumulator
            from pyspark.accumulators import AccumulatorParam

            class MatrixAccumulatorParam(AccumulatorParam):
                def zero(self, value):
                    return zeros(shape(value))

                def addInPlace(self, val1, val2):
                    val1 += val2
                    return val1

            # define an accumulator function
            global runSum

            def outerSumOther(x, y):
                global runSum
                runSum += outer(x, dot(x, y))

            # iterative update subspace using expectation maximization
            # e-step: x = (c'c)^-1 c' y
            # m-step: c = y x' (xx')^-1
            while (niter < self.maxIter) & (error > self.tol):

                cOld = c

                # pre compute (c'c)^-1 c'
                cInv = dot(c.T, inv(dot(c, c.T)))

                # compute (xx')^-1 through a map reduce
                xx = mat.times(cInv).gramian()
                xxInv = inv(xx)

                # pre compute (c'c)^-1 c' (xx')^-1
                preMult2 = mat.rdd.context.broadcast(dot(cInv, xxInv))

                # compute the new c using an accumulator
                # direct approach: c = mat.rows().map(lambda x: outer(x, dot(x, premult2.value))).sum()
                runSum = mat.rdd.context.accumulator(zeros((mat.ncols, self.k)), MatrixAccumulatorParam())
                mat.rows().foreach(lambda x: outerSumOther(x, preMult2.value))
                c = runSum.value

                # transpose result
                c = c.T

                error = sum(sum((c - cOld) ** 2))
                niter += 1

            # project data into subspace spanned by columns of c
            # use standard eigendecomposition to recover an orthonormal basis
            c = orth(c.T)
            cov = mat.times(c).gramian() / mat.nrows
            eigw, eigv = eigh(cov)
            inds = argsort(eigw)[::-1]
            s = sqrt(eigw[inds[0:self.k]]) * sqrt(mat.nrows)
            v = dot(eigv[:, inds[0:self.k]].T, c.T)
            u = mat.times(v.T / s)

            self.u = u
            self.s = s
            self.v = v

        return self
