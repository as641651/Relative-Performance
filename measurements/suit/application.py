from linnea.algebra.expression import Symbol, Scalar, Vector, Matrix, ConstantScalar, \
                                 Equal, Plus, Times, Transpose, Inverse, \
                                 InverseTranspose, InverseConjugate, \
                                 InverseConjugateTranspose, \
                                 ConjugateTranspose, Index, IdentityMatrix

from linnea.algebra.properties import Property

from linnea.algebra.equations import Equations

from linnea import derivation

from linnea.derivation import special_properties


class Example01():
    def __init__(self):

        # least squares

        n = 1000
        m = 500

        X = Matrix("X1", (n, m))
        X.set_property(Property.FULL_RANK)

        y = Vector("y1", (n, 1))

        b = Vector("b1", (m, 1))

        # b = (X^T X)^-1 X^T y
        self.eqns = Equations(Equal(b, Times(Inverse(Times(Transpose(X), X)), Transpose(X), y)))


class Example02():
    def __init__(self):

        # generalized least squares

        n = 250
        m = 500

        S = Matrix("S2", (n, n))
        S.set_property(Property.SPD)

        X = Matrix("X2", (n, m))
        X.set_property(Property.FULL_RANK)

        z = Vector("z2", (m, 1))

        y = Vector("y2", (n, 1))

        self.eqns = Equations(Equal(z, Times(Inverse(Times(Transpose(X), Inverse(S), X ) ), Transpose(X), Inverse(S), y)))


class Example03():
    def __init__(self, ):

        # optimization problem 1

        n = 200
        m = 100

        A = Matrix("A3", (m, n))
        A.set_property(Property.FULL_RANK)

        W = Matrix("W3", (n, n))
        # W is positive
        W.set_property(Property.FULL_RANK)
        W.set_property(Property.DIAGONAL)
        W.set_property(Property.SPD)

        b = Vector("b3", (m, 1))

        c = Vector("c3", (n, 1))

        x = Vector("x3", (n, 1))

        minus1 = ConstantScalar(-1.0)


        self.eqns = Equations(
                            Equal(
                                x,
                                Times(
                                    W,
                                    Plus(
                                        Times(Transpose(A), Inverse(Times(A, W, Transpose(A))), b),
                                        Times(minus1, c)
                                        )
                                    )
                                )
                            )


class Example04():
    def __init__(self, ):

        # optimization problem 2

        n = 200
        m = 100

        A = Matrix("A4", (m, n))
        A.set_property(Property.FULL_RANK)

        W = Matrix("W4", (n, n))
        # W is positive
        W.set_property(Property.FULL_RANK)
        W.set_property(Property.DIAGONAL)
        W.set_property(Property.SPD)

        b = Vector("b4", (m, 1))

        c = Vector("c4", (n, 1))

        x = Vector("x4", (n, 1))

        xf = Vector("xf4", (n, 1))

        xo = Vector("xo4", (n, 1))

        minus1 = ConstantScalar(-1.0)



        self.eqns = Equations(
                        Equal(xf,
                            Times(
                                W,
                                Transpose(A),
                                Inverse(Times(A, W, Transpose(A))),
                                Plus(b, Times(minus1, A, x))
                            )
                        ),
                        Equal(xo,
                            Times(W,
                                Plus(
                                    Times(
                                        Transpose(A),
                                        Inverse(Times(A, W, Transpose(A))),
                                        A, x),
                                    Times(minus1, c)
                                )
                            )
                        )
                    )


class Example05():
    def __init__(self):

        N = 1000

        # signal processing

        A = Matrix("A5", (N, N), properties = [Property.FULL_RANK])
        # A - tridiagonal, full rank, something even more specific
        B = Matrix("B5", (N, N), properties = [Property.FULL_RANK])
        # B - tridiagonal, full rank, something even more specific
        R = Matrix("R5", (N - 1, N), properties = [Property.FULL_RANK, Property.UPPER_TRIANGULAR])
        # R - upper bidiagonal
        L = Matrix("L5", (N - 1, N - 1), properties = [Property.FULL_RANK, Property.DIAGONAL])

        y = Vector("y5", (N, 1))
        x = Vector("x5", (N, 1))

        self.eqns = Equations(
                            Equal(
                                x,
                                Times(
                                    Inverse(
                                        Plus(
                                            Times(
                                                InverseTranspose(A),
                                                Transpose(B),
                                                B,
                                                Inverse(A)
                                            ),
                                            Times(
                                                Transpose(R),
                                                L,
                                                R
                                            )
                                        )
                                    ),
                                    InverseTranspose(A),
                                    Transpose(B),
                                    B,
                                    Inverse(A),
                                    y
                                )
                            )
                    )


class Example06():
    def __init__(self):

        # inversion of lower triangular matrix

        n = 400
        m = 200
        k = 400

        L00 = Matrix("L006", (n, n), properties = [Property.FULL_RANK, Property.LOWER_TRIANGULAR])
        L11 = Matrix("L116", (m, m), properties = [Property.FULL_RANK, Property.LOWER_TRIANGULAR])
        L22 = Matrix("L226", (k, k), properties = [Property.FULL_RANK, Property.LOWER_TRIANGULAR])
        L21 = Matrix("L216", (k, m), properties = [Property.FULL_RANK])
        L10 = Matrix("L106", (m, n), properties = [Property.FULL_RANK])
        L20 = Matrix("L206", (k, n), properties = [Property.FULL_RANK])

        X21 = Matrix("X216", (k, m))
        X11 = Matrix("X116", (m, m))
        X10 = Matrix("X106", (m, n))

        X20 = Matrix("X206", (k, n))
        minus1 = ConstantScalar(-1.0)

        self.eqns = Equations(
                            Equal(
                                X10,
                                Times(
                                    L10,
                                    Inverse(
                                        L00
                                    )
                                )
                            ),
                            Equal(
                                X20,
                                Plus(
                                    L20,
                                    Times(
                                        Inverse(L22),
                                        L21,
                                        Inverse(L11),
                                        L10
                                    )
                                )
                            ),
                            Equal(
                                X11,
                                Inverse(L11)
                            ),
                            Equal(
                                X21,
                                Times(
                                    minus1,
                                    Inverse(L22),
                                    L21
                                )
                            )
                    )


class Example07():
    def __init__(self):

        # local assimilation for parallel ensemble Kalman filter based on modified Cholesky decomposition 1

        N = 20
        msd = 200 # p*nsd, p < 1 (percentage)
        nsd = 200

        # TODO there is a dimension mismatch here for msd != nsd

        minus1 = ConstantScalar(-1.0)

        B = Matrix("B7", (nsd, nsd), properties = [Property.SPSD]) # covariance matrix
        H = Matrix("H7", (msd, nsd), properties = [Property.FULL_RANK])
        R = Matrix("R7", (msd, msd), properties = [Property.SPSD]) # covariance matrix
        Y = Matrix("Y7", (msd, N), properties = [Property.FULL_RANK])
        Xb = Matrix("Xb7", (nsd, N), properties = [Property.FULL_RANK])
        Xa = Matrix("Xa7", (nsd, N), properties = [Property.FULL_RANK])

        self.eqns = Equations(
                            Equal(
                                Xa,
                                Plus(
                                    Xb,
                                    Times(
                                        Inverse(Plus(Inverse(B), Times(Transpose(H), Inverse(R), H))),
                                        Plus(Y, Times(minus1, H, Xb))
                                        )
                                    )
                                )
                            )


class Example08():
    def __init__(self):

        # local assimilation for parallel ensemble Kalman filter based on modified Cholesky decomposition 2

        N = 20
        m = 100 # p*nsd, p < 1 (percentage)
        n = 500

        minus1 = ConstantScalar(-1.0)

        B = Matrix("B8", (n, n), properties = [Property.SPSD]) # covariance matrix
        H = Matrix("H8", (m, n), properties = [Property.FULL_RANK])
        R = Matrix("R8", (m, m), properties = [Property.SPSD]) # covariance matrix
        Y = Matrix("Y8", (m, N), properties = [Property.FULL_RANK])
        Xb = Matrix("Xb8", (n, N), properties = [Property.FULL_RANK])
        dX = Matrix("dX8", (n, N), properties = [Property.FULL_RANK])

        self.eqns = Equations(
                        Equal(dX,
                            Times(
                                Inverse(Plus(Inverse(B), Times(Transpose(H), Inverse(R), H))),
                                Transpose(H),
                                Inverse(R),
                                Plus(Y, Times(minus1, H, Xb))
                                )
                            )
                        )

class Example09():
    def __init__(self):

        # local assimilation for parallel ensemble Kalman filter based on modified Cholesky decomposition 3

        N = 20
        m = 100 # p*nsd, p < 1 (percentage)
        n = 500

        minus1 = ConstantScalar(-1.0)

        H = Matrix("H9", (m, n), properties = [Property.FULL_RANK])
        R = Matrix("R9", (m, m), properties = [Property.SPSD]) # covariance matrix
        Y = Matrix("Y9", (m, N), properties = [Property.FULL_RANK])
        X = Matrix("X9", (n, n), properties = [Property.FULL_RANK, Property.LOWER_TRIANGULAR])
        Xb = Matrix("Xb9", (n, N), properties = [Property.FULL_RANK])
        dX = Matrix("dX9", (n, N), properties = [Property.FULL_RANK])

        self.eqns = Equations(
                        Equal(dX,
                            Times(
                                X,
                                Transpose(Times(H, X)),
                                Inverse(Plus(R, Times(H, X, Transpose(Times(H, X))))),
                                Plus(Y, Times(minus1, H, Xb))
                                )
                            )
                        )


class Example10():
    def __init__(self):

        # image restoration 1

        n = 500
        m = 100

        minus1 = ConstantScalar(-1.0)
        lambda_ = Scalar("lambda10")
        lambda_.set_property(Property.POSITIVE)
        sigma_ = Scalar("sigma_sq10")
        sigma_.set_property(Property.POSITIVE)

        H = Matrix("H10", (m, n), properties = [Property.FULL_RANK])
        I = IdentityMatrix(n, n)

        v_k = Vector("v_k10", (n, 1))
        u_k = Vector("u_k10", (n, 1))
        y = Vector("y10", (m, 1))
        x = Vector("x10", (n, 1))


        self.eqns = Equations(
                            Equal(
                                x,
                                Times(
                                    # (H^t * H + lambda * sigma^2 * I_n)^-1
                                    Inverse( Plus(
                                        Times(
                                            Transpose(H),
                                            H
                                        ),
                                        Times(
                                            lambda_,
                                            sigma_,
                                            I
                                        )
                                    )),
                                    # (H^T * y + lambda * sigma^2 * (v - u))
                                    Plus(
                                        Times(
                                            Transpose(H),
                                            y
                                        ),
                                        Times(
                                            lambda_,
                                            sigma_,
                                            Plus(
                                                v_k,
                                                Times(
                                                    minus1,
                                                    u_k
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )


class Example11():
    def __init__(self):

        # image restoration 2

        n = 500
        m = 100

        minus1 = ConstantScalar(-1.0)
        lambda_ = Scalar("lambda11")
        lambda_.set_property(Property.POSITIVE)
        sigma_ = Scalar("sigma_sq11")
        sigma_.set_property(Property.POSITIVE)

        H = Matrix("H11", (m, n), properties = [Property.FULL_RANK])
        H_dag = Matrix("H_dag11", (n, m), properties = [Property.FULL_RANK])
        I = IdentityMatrix(n, n)

        y_k = Vector("y_k11", (n, 1))
        y = Vector("y11", (m, 1))
        x = Vector("x11", (n, 1))

        h_dag = Times(
                    Transpose(H),
                    Inverse(
                        Times(
                            H,
                            Transpose(H)
                        )
                    )
                )

        self.eqns = Equations(
                        Equal(
                            H_dag,
                            h_dag
                        ),
                        Equal(
                            y_k,
                            Plus(
                                Times(
                                    H_dag,
                                    y
                                ),
                                Times(
                                    Plus(
                                        I,
                                        Times(
                                            minus1,
                                            H_dag,
                                            H
                                        )
                                    ),
                                    x
                                )
                            )
                        )
                    )


class Example12():
    def __init__(self):

        # randomized matrix inversion 1

        # q << n
        n = 1000
        q = 100

        W = Matrix("W12", (n, n))
        W.set_property(Property.SPD)

        S = Matrix("S12", (n, q))
        S.set_property(Property.FULL_RANK)

        A = Matrix("A12", (n, n))
        A.set_property(Property.FULL_RANK)

        Lambda = Matrix("Lambda12", (n, n))
        Lambda.set_property(Property.SYMMETRIC)
        Lambda.set_property(Property.FULL_RANK)

        Xin = Matrix("Xin12", (n, n))
        Xin.set_property(Property.FULL_RANK)

        Xout = Matrix("Xout12", (n, n))

        I = IdentityMatrix(n, n)
        minus1 = ConstantScalar(-1.0)

        self.eqns = Equations(
                        Equal(Lambda,
                                Times(S, Inverse(Times(Transpose(S), A, W, Transpose(A), S)), Transpose(S))),
                        Equal(Xout,
                            Plus(Xin,
                                Times(
                                    W, Transpose(A), Lambda,
                                    Plus(I, Times(minus1, A, Xin))
                                    )
                                )
                            )
                        )

class Example13():
    def __init__(self):

        # randomized matrix inversion 2

        # q << n
        n = 1000
        q = 100

        W = Matrix("W13", (n, n))
        W.set_property(Property.SPD)

        S = Matrix("S13", (n, q))
        S.set_property(Property.FULL_RANK)

        A = Matrix("A13", (n, n))
        A.set_property(Property.FULL_RANK)

        Lambda = Matrix("Lambda13", (n, n))
        Lambda.set_property(Property.SYMMETRIC)
        Lambda.set_property(Property.FULL_RANK)

        Xin = Matrix("Xin13", (n, n))
        Xin.set_property(Property.FULL_RANK)

        Xout = Matrix("Xout13", (n, n))

        I = IdentityMatrix(n, n)
        minus1 = ConstantScalar(-1.0)

        self.eqns = Equations(
                        Equal(Lambda,
                            Times(S, Inverse(Times(Transpose(S), Transpose(A), W, A, S)), Transpose(S))),
                        Equal(Xout,
                            Plus(Xin,
                                Times(
                                    Plus(I, Times(minus1, Xin, Transpose(A))),
                                    Lambda, Transpose(A), W
                                    )
                                )
                            )
                        )


class Example14():
    def __init__(self):

        # randomized matrix inversion 3

        # q << n
        n = 1000
        q = 100

        W = Matrix("W14", (n, n))
        W.set_property(Property.SPD)

        S = Matrix("S14", (n, q))
        S.set_property(Property.FULL_RANK)

        A = Matrix("A14", (n, n))
        A.set_property(Property.SYMMETRIC)
        A.set_property(Property.FULL_RANK)

        Lambda = Matrix("Lambda14", (n, n))
        Lambda.set_property(Property.SYMMETRIC)
        Lambda.set_property(Property.FULL_RANK)

        Theta = Matrix("Theta14", (n, n))
        Theta.set_property(Property.FULL_RANK)

        Mk = Matrix("Mk14", (n, n))
        Mk.set_property(Property.FULL_RANK)

        Xin = Matrix("Xin14", (n, n))
        Xin.set_property(Property.SYMMETRIC)
        Xin.set_property(Property.FULL_RANK)

        Xout = Matrix("Xout14", (n, n))

        I = IdentityMatrix(n, n)
        minus1 = ConstantScalar(-1.0)


        self.eqns = Equations(
                        Equal(Lambda, Times(S, Inverse(Times(Transpose(S), A, W, A, S)), Transpose(S))),
                        Equal(Theta, Times(Lambda, A, W)),
                        Equal(Mk, Plus(Times(Xin, A), Times(minus1, I))),
                        Equal(Xout,
                            Plus(Xin,
                                Times(minus1, Mk, Theta),
                                Times(minus1, Transpose(Times(Mk, Theta))),
                                Times(
                                    Transpose(Theta),
                                    Plus(Times(A, Xin, A), Times(minus1, A)),
                                    Theta
                                    )
                                )
                            )
                        )


class Example15():
    def __init__(self):

        # randomized matrix inversion 4

        # q << n
        n = 2000
        q = 200

        S = Matrix("S15", (n, q))
        S.set_property(Property.FULL_RANK)

        A = Matrix("A15", (n, n))
        A.set_property(Property.SPD)
        A.set_property(Property.FULL_RANK)

        Xin = Matrix("Xin15", (n, n))
        Xin.set_property(Property.SYMMETRIC)
        Xin.set_property(Property.FULL_RANK)

        Xout = Matrix("Xout15", (n, n))
        Xout.set_property(Property.FULL_RANK)

        I = IdentityMatrix(n, n)
        minus1 = ConstantScalar(-1.0)

        subexpr = Times(S, Inverse(Times(Transpose(S), A, S)), Transpose(S))

        self.eqns = Equations(
                        Equal(Xout,
                            Plus(
                                subexpr,
                                Times(
                                    Plus(I, Times(minus1, subexpr, A)),
                                    Xin,
                                    Plus(I, Times(minus1, A, subexpr))
                                    )
                                )
                            )
                        )


class Example16():
    def __init__(self):

        # Stochastic Newton and Quasi-Newton for Large Linear Least-squares 1

        # l < n
        # n << m
        l = 62
        n = 100
        m = 500

        Wk = Matrix("Wk16", (m, l))
        Wk.set_property(Property.FULL_RANK)

        A = Matrix("A16", (m, n))
        A.set_property(Property.FULL_RANK)

        Bin = Matrix("Bin16", (n, n))
        Bin.set_property(Property.SPD)

        Bout = Matrix("Bout16", (n, n))
        Bout.set_property(Property.SPD)

        In = IdentityMatrix(n, n)
        Il = IdentityMatrix(l, l)
        k = Scalar("k16")
        k.set_property(Property.POSITIVE)
        minus1 = ConstantScalar(-1.0)
        kminus1 = Plus(k, minus1) # this is positive too, but using special_properties doesn't work

        # This works:
        # special_properties.add_expression(Plus(Times(kminus1, Il), Times(Transpose(Wk), A, Bin, Transpose(A), Wk)), [Property.SPD])

        self.eqns = Equations(
                        Equal(Bout,
                            Times(
                                Times(k, Inverse(kminus1)),
                                Bin,
                                Plus(
                                    In,
                                    Times(
                                        minus1, Transpose(A), Wk,
                                        Inverse(Plus(
                                            Times(kminus1, Il),
                                            Times(Transpose(Wk), A, Bin, Transpose(A), Wk)
                                            )),
                                        Transpose(Wk), A, Bin
                                        )
                                    )
                                )
                            )
                        )


class Example17():
    def __init__(self):

        # Stochastic Newton and Quasi-Newton for Large Linear Least-squares 2

        # l < n
        # n << m
        l = 62
        n = 100
        m = 500

        Wk = Matrix("Wk17", (m, l))
        Wk.set_property(Property.FULL_RANK)

        A = Matrix("A17", (m, n))
        A.set_property(Property.FULL_RANK)

        B = Matrix("B17", (n, n))
        B.set_property(Property.SPD)

        In = IdentityMatrix(n, n)
        Il = IdentityMatrix(l, l)
        lambda_ = Scalar("lambda17")
        lambda_.set_property(Property.POSITIVE)
        minus1 = ConstantScalar(-1.0)

        self.eqns = Equations(
                        Equal(B,
                            Times(
                                Times(ConstantScalar(1.0), Inverse(lambda_)),
                                Plus(
                                    In,
                                    Times(
                                        minus1, Transpose(A), Wk,
                                        Inverse(Plus(
                                            Times(lambda_, Il),
                                            Times(Transpose(Wk), A, Transpose(A), Wk)
                                            )),
                                        Transpose(Wk), A
                                        )
                                    )
                                )
                            )
                        )


class Example18():
    def __init__(self):

        # tikhonov regularization 1

        n = 300
        m = 200

        A = Matrix("A18", (n, m), properties = [Property.FULL_RANK])
        Gamma = Matrix("Gamma18", (m , m), properties = [Property.FULL_RANK])
        b = Vector("b18", (n, 1))
        x = Vector("x18", (m, 1))

        self.eqns = Equations(
                            Equal(x,
                                Times(Inverse(Plus(Times(Transpose(A), A), Times(Transpose(Gamma), Gamma))), Transpose(A), b)
                                )
                            )


class Example19():
    def __init__(self):

        # tikhonov regularization 2

        n = 300
        m = 200

        A = Matrix("A19", (n, m), properties = [Property.FULL_RANK])
        I = IdentityMatrix(m, m)
        b = Vector("b19", (n, 1))
        x = Vector("x19", (m, 1))
        alpha = Scalar("alpha_sq19")
        alpha.set_property(Property.POSITIVE)

        self.eqns = Equations(
                            Equal(x,
                                Times(Inverse(Plus(Times(Transpose(A), A), Times(alpha, I))), Transpose(A), b)
                                )
                            )


class Example20():
    def __init__(self):

        # tikhonov regularization 3

        n = 300
        m = 200

        A = Matrix("A20", (n, m), properties = [Property.FULL_RANK])
        P = Matrix("P20", (n, n), properties = [Property.SPD]) # covariance matrix
        Q = Matrix("Q20", (m, m), properties = [Property.SPD]) # covariance matrix
        b = Vector("b20", (n, 1))
        x0 = Vector("x020", (m, 1))
        x = Vector("x20", (m, 1))

        self.eqns = Equations(
                            Equal(x,
                                Times(
                                    Inverse(Plus(Times(Transpose(A), P, A), Q)),
                                    Plus(Times(Transpose(A), P, b), Times(Q, x0))
                                    )
                                )
                            )


class Example21():
    def __init__(self):

        # tikhonov regularization 4

        n = 300
        m = 200

        A = Matrix("A21", (n, m), properties = [Property.FULL_RANK])
        P = Matrix("P21", (n, n), properties = [Property.SPD]) # covariance matrix
        Q = Matrix("Q21", (m, m), properties = [Property.SPD]) # covariance matrix
        b = Vector("b21", (n, 1))
        x0 = Vector("x021", (m, 1))
        x = Vector("x21", (m, 1))
        minus1 = ConstantScalar(-1.0)

        self.eqns = Equations(
                            Equal(x,
                                Plus(x0,
                                    Times(
                                        Inverse(Plus(Times(Transpose(A), P, A), Q)),
                                        Transpose(A), P, Plus(b, Times(minus1, A, x0))
                                        )
                                    )
                                )
                            )


class Example22():
    def __init__(self):

        # linear MMSE estimator 1

        n = 200
        m = 150

        A = Matrix("A22", (m, n), properties = [Property.FULL_RANK])
        Cx = Matrix("Cx22", (n, n), properties = [Property.SPD]) # covariance matrix
        Cz = Matrix("Cz22", (m, m), properties = [Property.SPD]) # covariance matrix
        y = Vector("y22", (m, 1))
        x = Vector("x22", (n, 1))
        xout = Vector("xout22", (n, 1))
        minus1 = ConstantScalar(-1.0)

        self.eqns = Equations(
                        Equal(xout,
                            Plus(
                                Times(Cx, Transpose(A),
                                    Inverse(Plus(Times(A, Cx, Transpose(A)), Cz)),
                                    Plus(y, Times(minus1, A, x))
                                    ),
                                x)
                            )
                        )


class Example23():
    def __init__(self):

        # linear MMSE estimator 2

        n = 1000
        m = 500

        A = Matrix("A23", (m, n), properties = [Property.FULL_RANK])
        Cx = Matrix("Cx23", (n, n), properties = [Property.SPSD]) # covariance matrix
        Cz = Matrix("Cz23", (m, m), properties = [Property.SPSD]) # covariance matrix
        y = Vector("y23", (m, 1))
        x = Vector("x23", (n, 1))
        xout = Vector("xout23", (n, 1))
        minus1 = ConstantScalar(-1.0)

        self.eqns = Equations(
                        Equal(xout,
                            Plus(
                                Times(
                                    Inverse(Plus(Times(Transpose(A), Inverse(Cz), A), Inverse(Cx))),
                                    Transpose(A), Inverse(Cz),
                                    Plus(y, Times(minus1, A, x))
                                    ),
                                x)
                            )
                        )


class Example24():
    def __init__(self):

        # linear MMSE estimator 3

        n = 1000
        m = 500

        A = Matrix("A24", (m, n), properties = [Property.FULL_RANK])
        K = Matrix("K24", (n, m), properties = [Property.FULL_RANK])
        Cin = Matrix("Cin24", (n, n), properties = [Property.SPD]) # covariance matrix
        Cz = Matrix("Cz24", (m, m), properties = [Property.SPD]) # covariance matrix
        y = Vector("y24", (m, 1))
        x = Vector("x24", (n, 1))

        xout = Vector("xout24", (n, 1))
        Cout = Matrix("Cout24", (n, n)) # covariance matrix

        minus1 = ConstantScalar(-1.0)
        I = IdentityMatrix(n, n)

        self.eqns = Equations(
                        Equal(K,
                            Times(Cin, Transpose(A), Inverse(Plus(Times(A, Cin, Transpose(A)), Cz)))),
                        Equal(xout,
                            Plus(x, Times(K, Plus(y, Times(minus1, A, x))))),
                        Equal(Cout,
                            Times(Plus(I, Times(minus1, K, A)), Cin))
            )

class Example25():
    def __init__(self):

        # Kalman filter

        n = 400
        m = 400
        minus1 = ConstantScalar(-1.0)

        Kk = Matrix("Kk25", (n, m), properties = [Property.FULL_RANK])
        P_b = Matrix("P_b25", (n, n), properties = [Property.SPD])
        P_a = Matrix("P_a25", (n, n))
        H = Matrix("H25", (m, n), properties = [Property.FULL_RANK])
        R = Matrix("R25", (m, m), properties = [Property.SPSD]) # covariance matrix
        I = IdentityMatrix(n, n)

        x_a = Vector("x_a25", (n, 1))
        x_b = Vector("x_b25", (n, 1))
        zk = Vector("zk25", (m, 1))

        self.eqns = Equations(
                            Equal(
                                Kk,
                                Times(
                                    P_b,
                                    Transpose(H),
                                    Inverse(Plus(Times(H, P_b, Transpose(H)), R))
                                )
                            ),
                            Equal(x_a, Plus(x_b, Times(Kk, Plus( zk, Times(minus1, H, x_b))))),
                            Equal(P_a, Times(Plus(I, Times(minus1, Kk, H)), P_b))
                        )
