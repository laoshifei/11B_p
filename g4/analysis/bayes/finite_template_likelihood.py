"""Conditional data + fixed-generated-count Monte Carlo likelihood.

Each pure MC channel has a binomial acceptance and a multinomial accepted
shape. Data are conditioned on their accepted count, exactly as in the
previous study. Shapes and efficiencies are frequentist nuisance parameters;
no pseudocounts enter this likelihood and nothing is Bayesian-marginalized.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import xlogy

VERSION = "2026-10-22-v1-joint-multinomial"


def accepted_weights(fractions, efficiency):
    value = np.asarray(fractions)*np.asarray(efficiency)
    return value/value.sum()


def generated_fractions(weights, efficiency):
    value = np.asarray(weights)/np.asarray(efficiency)
    return value/value.sum()


def check_stationarity(objective, parameters, bounds, *, gradient_tolerance=5e-4):
    """Check projected score, then Newton decrement if score units are stiff.

    The decrement is a local numerical convergence check, not a global
    optimality theorem. The inner shape optimization has its separate bound.
    """
    parameters = np.asarray(parameters)
    _, gradient = objective(parameters)
    free = []
    projected = gradient.copy()
    for i, (low, high) in enumerate(bounds):
        if (parameters[i] <= low+1e-8 and gradient[i] >= 0
                or parameters[i] >= high-1e-8 and gradient[i] <= 0):
            projected[i] = 0.
        else:
            free.append(i)
    maximum = float(np.max(abs(projected)))
    if maximum <= gradient_tolerance or not free:
        return maximum, 0.
    hessian = np.empty((len(free), len(free)))
    for column, index in enumerate(free):
        low, high = bounds[index]
        left, right = parameters.copy(), parameters.copy()
        left[index], right[index] = max(low, parameters[index]-.02), min(high, parameters[index]+.02)
        hessian[:, column] = (objective(right)[1][free]-objective(left)[1][free])/(right[index]-left[index])
    hessian = (hessian+hessian.T)/2
    if np.linalg.eigvalsh(hessian).min() <= 0:
        raise RuntimeError(f"Nonpositive local profile curvature; projected score {projected}")
    decrement = float(.5*projected[free] @ np.linalg.solve(hessian, projected[free]))
    if decrement > 2e-4:
        raise RuntimeError(f"Profile optimum needs refinement: score={maximum:g}, Newton decrement={decrement:g}")
    return maximum, decrement


class JointMultinomial:
    """Profile the exact finite-MC multinomial likelihood on declared bins.

    For fixed accepted weights, shape maximization is a concave problem on
    three simplices. EM is stopped using its concave first-order upper bound,
    not just a small parameter update. Empty MC bins may acquire probability.
    """

    def __init__(self, data, templates, generated, *, start=None, tolerance=1e-4):
        self.n = np.asarray(data, float)
        self.m = np.asarray(templates, float)
        self.generated = np.asarray(generated, float)
        if (self.n.ndim != 1 or self.m.shape != (3, len(self.n))
                or self.generated.shape != (3,) or not np.isfinite(self.n).all()
                or not np.isfinite(self.m).all() or not np.isfinite(self.generated).all()
                or np.any(self.n < 0) or np.any(self.m < 0) or self.n.sum() <= 0
                or np.any(self.generated <= 0) or tolerance <= 0):
            raise ValueError("Invalid data or fixed-total MC counts")
        self.total = self.n.sum()
        self.accepted = self.m.sum(axis=1)
        self.rejected = self.generated-self.accepted
        if np.any(self.accepted <= 0) or np.any(self.rejected < 0):
            raise ValueError("Every MC channel needs accepted counts within its generated budget")
        self.efficiency = self.accepted/self.generated
        self.tolerance = float(tolerance)
        self.reference_shape = self.m/self.accepted[:, None]
        self.relative = self.n/self.total
        self.observed = self.n > 0
        self._cache = OrderedDict()
        self._warm = None
        self.max_shape_iterations = 0
        self.max_shape_gap = 0.
        start = np.array([.05, .94, .01]) if start is None else np.asarray(start, float)
        if start.shape != (3,) or np.any(start < 0) or not np.isclose(start.sum(), 1):
            raise ValueError("Invalid accepted-weight initial guess")
        self._find_mode(start)

    def shape(self, weights):
        weights = np.asarray(weights, float)
        if (weights.shape != (3,) or np.any(weights < -1e-12)
                or not np.isclose(weights.sum(), 1., atol=1e-12)):
            raise ValueError("Invalid accepted weights")
        weights = np.maximum(weights, 0.)
        weights /= weights.sum()
        key = tuple(weights)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        shapes = (self.m+.1)/(self.accepted[:, None]+.1*len(self.n)) if self._warm is None else self._warm.copy()
        # Reseed sampling zeros when the active channel set changes. These
        # values only initialize EM; they are not auxiliary MC observations.
        shapes[:, self.observed] = np.maximum(shapes[:, self.observed], 1e-12)
        shapes /= shapes.sum(axis=1, keepdims=True)

        def step(current):
            q = weights @ current
            ratio = np.divide(self.n, q, out=np.zeros_like(q), where=q > 0)
            if np.any((q <= 0) & self.observed):
                raise RuntimeError("Positive data have no fitted support")
            latent = current*(weights[:, None]*ratio)
            return (self.m+latent)/(self.accepted+latent.sum(axis=1))[:, None]

        def loss_at(current):
            q = weights @ current
            if np.any((q <= 0) & self.observed) or np.any((current <= 0) & (self.m > 0)):
                return np.inf
            positive_mc = self.m > 0
            return float(np.sum(xlogy(self.n[self.observed], self.relative[self.observed]/q[self.observed]))
                         +np.sum(xlogy(self.m[positive_mc], self.reference_shape[positive_mc]/current[positive_mc])))

        for iteration in range(1, 1001):
            first = step(shapes)
            second = step(first)
            change = first-shapes
            curvature = second-first-change
            denominator = float(np.sum(curvature*curvature))
            alpha = (-min(100., max(1., np.sqrt(np.sum(change*change)/denominator)))
                     if denominator > 1e-300 else -1.)
            reference_loss = loss_at(second)
            candidate = second
            # Accelerate slow, sparse-bin fixed points, retaining a monotone
            # likelihood. Concavity-gap stopping still certifies the result.
            for _ in range(12):
                proposed = shapes-2*alpha*change+alpha*alpha*curvature
                proposed = np.maximum(proposed, 0.)
                proposed[:, self.observed] = np.maximum(proposed[:, self.observed], 1e-300)
                proposed /= proposed.sum(axis=1, keepdims=True)
                accelerated = step(proposed)
                if loss_at(accelerated) <= reference_loss+1e-9:
                    candidate = accelerated
                    break
                alpha = (alpha-1)/2
            shapes = candidate
            q = weights @ shapes
            ratio = np.divide(self.n, q, out=np.zeros_like(q), where=q > 0)
            gradient = np.divide(self.m, shapes, out=np.zeros_like(shapes), where=shapes > 0)
            gradient += weights[:, None]*ratio
            multiplier = self.accepted+weights*(shapes @ ratio)
            gap = max(0., float(np.sum(gradient.max(axis=1)-multiplier)))
            if gap <= self.tolerance:
                break
            if iteration % 5 == 0:
                # A sampling-zero coordinate may sit at a tiny probability
                # with a residual score, although its likelihood change is
                # below floating-point EM progress. Release it additively
                # using its local curvature, then retain the row constraint.
                violation = gradient-multiplier[:, None]
                active_zero = (self.m == 0) & self.observed[None, :] & (violation > self.tolerance/3)
                if np.any(active_zero):
                    ratio_zero = np.divide(weights[:, None], q[None, :], out=np.zeros_like(shapes),
                                           where=self.observed[None, :] & (q[None, :] > 0))
                    curvature_zero = ratio_zero**2*self.n
                    curvature_zero += multiplier[:, None]
                    increment = np.where(active_zero, np.minimum(.001, np.maximum(violation,0)/curvature_zero), 0.)
                    proposed = shapes+increment
                    proposed /= proposed.sum(axis=1, keepdims=True)
                    if loss_at(proposed) <= loss_at(shapes)+1e-8:
                        shapes = proposed
        else:
            raise RuntimeError(f"Shape profiling did not converge: gap={gap:g}, weights={weights}")
        q = weights @ shapes
        loss = loss_at(shapes)
        score = shapes @ np.divide(self.n, q, out=np.zeros_like(q), where=q > 0)
        result = dict(loss=loss, gradient=score, shapes=shapes, q=q, gap=gap, iterations=iteration)
        self._warm = shapes
        self.max_shape_iterations = max(self.max_shape_iterations, iteration)
        self.max_shape_gap = max(self.max_shape_gap, gap)
        self._cache[key] = result
        if len(self._cache) > 24:
            self._cache.popitem(last=False)
        return result

    def _find_mode(self, start):
        def null_objective(a):
            return self.shape([a, 1-a, 0.])["loss"]
        def score(a):
            g = self.shape([a, 1-a, 0.])["gradient"]
            return g[0]-g[1]
        centre = float(start[0]/max(1e-15, start[0]+start[1]))
        value = score(centre)
        low = high = centre
        step_size = .01
        if value > 0:
            while high < 1:
                high = min(1., centre+step_size)
                if score(high) <= 0:
                    break
                step_size *= 2
        else:
            while low > 0:
                low = max(0., centre-step_size)
                if score(low) >= 0:
                    break
                step_size *= 2
        a_null = (brentq(score, low, high, xtol=1e-12) if low < high and score(low)*score(high) < 0
                  else min((null_objective(0.), 0.), (null_objective(1.), 1.))[1])
        candidates = [(null_objective(a_null), a_null), (null_objective(0.), 0.), (null_objective(1.), 1.)]
        self.null_loss, a_null = min(candidates)
        self.null_weights = np.array([a_null, 1-a_null, 0.])

        scale = .005
        def objective(parameters):
            a, v = parameters*scale
            weights = np.array([(1-v)*a, (1-v)*(1-a), v])
            result = self.shape(weights)
            g = result["gradient"]
            derivative = np.array([(1-v)*(g[0]-g[1]), g[2]-a*g[0]-(1-a)*g[1]])
            return result["loss"], -scale*derivative
        initial = np.array([a_null, max(.001, start[2])])/scale
        optimum = minimize(objective, initial, jac=True, method="SLSQP",
            bounds=[(0., 1/scale)]*2, options={"ftol": 2e-10, "maxiter": 100})
        a, v = optimum.x*scale
        weights = np.array([(1-v)*a, (1-v)*(1-a), v])
        best = self.shape(weights)
        if self.null_loss <= best["loss"]+1e-8:
            weights, best = self.null_weights.copy(), self.shape(self.null_weights)
        gap = max(0., float(best["gradient"].max()-weights @ best["gradient"]))
        parameters = np.array([weights[0]/max(1e-15,1-weights[2]), weights[2]])/scale
        self.mode_score, self.mode_decrement = check_stationarity(objective, parameters, [(0.,1/scale)]*2)
        self.weights, self.loss, self.mode_gap = weights, best["loss"], gap
        self.mode = generated_fractions(weights, self.efficiency)
        self.q0 = max(0., 2*(self.null_loss-self.loss))
        self._profile_cache = {0.: self.null_loss}

    def efficiency_loss(self, efficiency):
        return float(np.sum(xlogy(self.accepted, self.efficiency)-xlogy(self.accepted, efficiency)
                            +xlogy(self.rejected, 1-self.efficiency)-xlogy(self.rejected, 1-efficiency)))

    def profile_direct(self, direct, *, vary_efficiency=True):
        """Profile generator d, including MC binomial acceptance uncertainty."""
        direct = float(direct)
        if not 0 <= direct <= 1:
            raise ValueError("Fraction must be in [0,1]")
        if direct == 0:
            return self.null_loss
        if direct == 1:
            return self.shape([0., 0., 1.])["loss"]
        if vary_efficiency and direct in self._profile_cache:
            return self._profile_cache[direct]
        rstart = self.mode[0]/max(1e-15, 1-self.mode[2])
        rscale = .005
        escale = np.maximum(np.sqrt(self.efficiency*(1-self.efficiency)/self.generated), 1/self.generated)
        lower = (1e-12-self.efficiency)/escale
        upper = ((1-np.where(self.rejected > 0, 1e-12, 0.))-self.efficiency)/escale

        def objective(parameters):
            r = rstart+rscale*parameters[0]
            efficiency = self.efficiency+escale*parameters[1:] if vary_efficiency else self.efficiency
            efficiency = np.clip(efficiency, 1e-12, 1.)
            fractions = np.array([r*(1-direct), (1-r)*(1-direct), direct])
            weights = accepted_weights(fractions, efficiency)
            result = self.shape(weights)
            score = result["gradient"]
            mean_score = weights @ score
            derivative_r = np.array([1-direct, -(1-direct), 0.])*efficiency
            derivative_r = (derivative_r-weights*derivative_r.sum())/(fractions @ efficiency)
            gradient = [-rscale*(score @ derivative_r)]
            loss = result["loss"]
            if vary_efficiency:
                loss += self.efficiency_loss(efficiency)
                eff_score = self.accepted/efficiency-np.divide(self.rejected, 1-efficiency,
                    out=np.zeros(3), where=efficiency < 1)
                eff_score += fractions/(fractions @ efficiency)*(score-mean_score)
                gradient.extend(-escale*eff_score)
            return loss, np.asarray(gradient)

        dimensions = 4 if vary_efficiency else 1
        bounds = [(-rstart/rscale, (1-rstart)/rscale)]
        if vary_efficiency:
            bounds.extend(zip(lower, upper))
        optimum = minimize(objective, np.zeros(dimensions), jac=True, method="SLSQP", bounds=bounds,
                           options={"ftol": 2e-9, "maxiter": 100})
        value, gradient = objective(optimum.x)
        check_stationarity(objective, optimum.x, bounds)
        if value < self.loss-4e-4:
            raise RuntimeError(f"Generator profile exceeds the mixture maximum: {value-self.loss:g}")
        if vary_efficiency:
            self._profile_cache[direct] = value
        return value

    def statistic(self, direct, *, upper=False):
        if upper and self.mode[2] > direct:
            return 0.
        return max(0., 2*(self.profile_direct(direct)-self.loss))

    def upper_limit(self, cutoff):
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Positive finite cutoff required")
        centre = float(self.mode[2])
        if self.statistic(1.) <= cutoff:
            return 1.
        high = min(1., max(.01, centre+.01))
        while self.statistic(high) < cutoff and high < 1:
            high = min(1., centre+2*(high-centre))
        return brentq(lambda d: self.statistic(d)-cutoff, centre, high, xtol=2e-8)
