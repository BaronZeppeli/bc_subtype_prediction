import numpy as np
from scipy.optimize import minimize


class PositiveLinearEnsembler:
    """
    For use in ensemble combination: Fits a positive, sum-to-1 linear
    combination of model output by minimizing binary cross-entropy.
    """

    def __init__(self,
                 reg_l2: float = 0.0,
                 eps: float = 1e-12,
                 maxiter: int = 1000,
                 ftol: float = 1e-9):
        """
        reg: L2 regularization strength
        eps: numerical epsilon for clipping
        """
        self.reg_l2 = reg_l2
        self.eps = eps
        self.maxiter = maxiter
        self.ftol = ftol
        self.weights_ = None

    # ----- LOSS -----
    def _loss(self,
              w: np.ndarray,
              P: np.ndarray,
              y: np.ndarray, ):
        s = P.dot(w)
        s = np.clip(s, self.eps, 1 - self.eps)
        loss = -np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))
        if self.reg_l2 > 0:
            loss += 0.5 * self.reg_l2 * np.sum(w ** 2)
        return loss

    # ----- GRADIENT -----
    def _gradient(self,
                  w: np.ndarray,
                  P: np.ndarray,
                  y: np.ndarray, ):
        s = P.dot(w)
        s = np.clip(s, self.eps, 1 - self.eps)
        dL_ds = -(y / s) + ((1 - y) / (1 - s))
        grad = P.T.dot(dL_ds)
        if self.reg_l2 > 0:
            grad += self.reg_l2 * w
        return grad

    # ----- FIT -----
    def fit(self,
            P: np.ndarray,
            y: np.ndarray):
        P = np.array(P)
        y = np.array(y)
        n_models = P.shape[1]

        w0 = np.ones(n_models) / n_models  # start with uniform weights

        # equality constraint: sum(w) = 1
        constraints = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1,
            'jac': lambda w: np.ones_like(w)
        }]

        # bounds: w_i >= 0
        bounds = [(0.0, None)] * n_models

        res = minimize(
            fun=lambda w: self._loss(w, P, y),
            x0=w0,
            jac=lambda w: self._gradient(w, P, y),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.maxiter, 'ftol': self.ftol}
        )

        if not res.success:
            raise RuntimeError(f"Optimizer failed: {res.message}")

        w_opt = np.clip(res.x, 0.0, None)
        assert np.all(res.x == w_opt), "Clipping did something weird."
        w_opt /= np.sum(w_opt)

        self.weights_ = w_opt
        return self

    # ----- PREDICT -----
    def predict(self,
                P: np.ndarray):
        if self.weights_ is None:
            raise ValueError("Model not fitted yet.")
        assert np.all(
            P.dot(self.weights_) <= 1), "Somehow clipping was required."
        return np.clip(P.dot(self.weights_), self.eps, 1 - self.eps)
