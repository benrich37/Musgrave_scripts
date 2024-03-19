
from numba import jit
import numpy as np
from sklearn.mixture import GaussianMixture

@jit(nopython=True)
def _smooth(y, nsamples, out):
    for i in range(1, nsamples-1):
        out[i] += np.average(y[i-1:i+2])
    out[0] += y[0]
    out[-1] += y[-1]
    return out

def smooth(y, loops = 1, _loop_cur = 0):
    nsamples = len(y)
    if _loop_cur < loops:
        out = np.zeros(nsamples)
        out = _smooth(y, nsamples, out)
        return smooth(out, loops = loops, _loop_cur = _loop_cur + 1)
    else:
        return y


def gaussfit(xx, y, n_gaussians, max_iter = 100, n_init = 1, tol  = 1e-3):
    y_normalized = y / y.sum()
    x_sample = np.repeat(xx, (y_normalized * 5000).astype(int))
    X = x_sample[:, np.newaxis]
    gmm = GaussianMixture(n_components=n_gaussians, max_iter=max_iter, tol=tol, n_init = n_init)
    gmm.fit(X)
    return gmm

def get_each_gaus(gmm, xrange):
    ngaus = len(gmm.weights_)
    yys = []
    for i in range(ngaus):
        yys.append(get_gaus_i(gmm, xrange, i))
    return yys


def get_gaus_i(gmm, xrange, i):
    weight_i = gmm.weights_[i]
    # covariances_ is a (n, 1, 1) array for some reason, thus the [0][0] is needed
    sq_cov_i = np.sqrt(gmm.covariances_[i][0][0])
    # means is a (n, 1) array for some reason, thus the [0] is needed
    means_i = gmm.means_[i][0]
    prefac = weight_i/sq_cov_i
    out = np.exp(-(1/2)*(
            (xrange - means_i)/(sq_cov_i)
    )**2)
    # Changed bc wierd output shape
    if len(np.shape(out)) > 1:
        return (prefac * out)[:, 0]
    else:
        return prefac * out

@jit(nopython=True)
def _get_dist_overlap(dist1, dist2, nsamples, _overlap = 0):
    for i in range(nsamples):
        _overlap += max(dist1[i], dist2[i]) * (min(dist1[i], dist2[i]) / max(dist1[i], dist2[i]))
    return _overlap

def get_dist_overlap(dist1, dist2, weight1, weight2):
    nsamples = len(dist1)
    _overlap = _get_dist_overlap(dist1, dist2, nsamples, _overlap = 0)
    return _overlap / (weight1 + weight2)

def get_overlap(gmm, xrange, i1, i2):
    dist1 = get_gaus_i(gmm, xrange, i1)
    dist2 = get_gaus_i(gmm, xrange, i2)
    weight1 = gmm.weights_[i1]
    weight2 = gmm.weights_[i2]
    return get_dist_overlap(dist1, dist2, weight1, weight2)

@jit(nopython=True)
def _find_closest_index(xval, xrange, _dist = 0.0, _best_dist = 1000.0, _idx = 0):
    for i in range(len(xrange)):
        _dist = xval - xrange[i]
        if _dist < 0:
            _dist *= -1
        if _dist < _best_dist:
            _best_dist = _dist
            _idx = i
    return _idx


def find_closest_index(xval, xrange):
    if len(np.shape(xrange)) > 1:
        xrange = xrange[:, 0]
    return _find_closest_index(xval, xrange)

@jit(nopython=True)
def _find_strattle_indices(xval, xrange, _best1 = 1000.0, _best2 = 1000.0, _idx1 = 0, _idx2 = 0, _dist = 0.0):
    for i in range(len(xrange)):
        _dist = xval - xrange[i]
        if _dist <= 0:
            _dist *= (-1.)
            if _dist < _best1:
                _best1 = _dist
                _idx1 = i
        else:
            if _dist < _best2:
                _best2 = _dist
                _idx2 = i
    return _idx1, _idx2

def find_strattle_indices(xval, xrange):
    if len(np.shape(xrange)) > 1:
        xrange = xrange[:, 0]
    arg1 = np.float64(xval)
    arg2 = np.array(xrange, dtype=np.float64)
    return _find_strattle_indices(arg1, arg2)

def recast_axis(xrange, xrange_old, y_old):
    assert(len(xrange_old) == len(y_old))
    ynew = np.zeros(len(xrange), dtype=np.float64)
    interp = len(xrange) < len(xrange_old)
    if len(np.shape(xrange)) > 1:
        xrange = xrange[:, 0]
    if interp:
        for i in range(len(xrange)):
            ynew[i] += y_old[find_closest_index(xrange[i], xrange_old)]
    else:
        for i in range(len(xrange)):
            i1, i2 = find_strattle_indices(xrange[i], xrange_old)
            ynew[i] += (y_old[i1] + y_old[i2])/2.
    return ynew

#@jit(nopython=True)
def get_distribution(yys, yy_sum_in):
    for yy in yys:
        yy_sum_in += yy
    return yy_sum_in

def get_score(xx, y, gmm):
    xx = np.array(xx, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    samples = 1000
    _yy_in=np.zeros(samples)
    xx_range = np.linspace(xx.min(), xx.max(), samples, dtype=np.float64)[:, np.newaxis]
    yys = get_each_gaus(gmm, xx_range)
    yy = get_distribution(yys, _yy_in)
    y = recast_axis(xx_range, xx, y)
    shift = np.max(y)/np.max(yy)
    yy *= shift
    off = 0
    for i in range(samples):
        off += abs(yy[i] - y[i])
    return off/(samples*(y.max()-y.min()))


def get_gauss_fit(xs, ys, nGauss = 15, fit_tol=1e-5, n_init=15, nSamples=1000, ysmooth_loops=0, print_score=True):
    xx = xs
    y = smooth(ys, loops=ysmooth_loops)
    gmm = gaussfit(xx, y, nGauss, tol=fit_tol, n_init=n_init)
    xx_range = np.linspace(xx.min(), xx.max(), nSamples)[:, np.newaxis]
    yy_gmm = np.exp(gmm.score_samples(xx_range))
    shift1 = np.max(y)/np.max(yy_gmm)
    yy_sum = np.zeros(len(xx_range))
    for yy in get_each_gaus(gmm, xx_range):
        yy_sum += yy
    if print_score:
        score = get_score(xx,y, gmm)
        print(f"Score: {score}")
    return xx_range, yy_gmm*shift1, gmm