# py_fractreg/py_fractreg.py
# exec(open('py_fractreg/py_fractreg/py_fractreg.py').read())
# 0.0.1

import numpy, scipy
from statsmodels.base.model import GenericLikelihoodModel as gll
from statsmodels.api import Logit as logit


#################### 01. Beta Regression: Fixed Dispersion ####################


def _ll_beta0(y, x, beta, ln_phi):
  """
  The function calculates the log likelihood function of a beta regression.
  Parameters:
    y      : the fractional outcome in the range of (0, 1)
    x      : variables of the location parameter in beta regression
    beta   : coefficients of the location parameter regression
    ln_phi : the dispersion parameter 
  """

  mu = 1 / (1 + numpy.exp(-numpy.dot(x, beta)))
  ph = numpy.exp(ln_phi)
  _w = mu * ph
  _t = (1 - mu) * ph
  pr = scipy.special.gamma(_w + _t) * numpy.float_power(y, _w - 1) * numpy.float_power(1 - y, _t - 1) \
       / scipy.special.gamma(_w) / scipy.special.gamma(_t)
  ll = numpy.log(pr)
  return(ll)


def beta0_reg(Y, X):
  """
  The function estimates a beta regression, assuming the fixed dispersion.
  In the model output, coefficients are estimated for the location parameter.
  Parameters:
    Y : a pandas series for the fractional outcome in the range of (0, 1)
    X : a pandas dataframe with the location model variables that are all numeric values.
  Example:
    beta0_reg(Y, X).fit().summary()   
  """

  class _beta0(gll):
    def __init__(self, endog, exog, **kwds):
      super(_beta0, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      beta = params[:-1]
      ln_phi  = params[-1]
      ll = _ll_beta0(self.endog, self.exog, beta, ln_phi)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_LN_PHI')
      if start_params == None:
        start_params = numpy.append(p10, p20)
      return(super(_beta0, self).fit(start_params = start_params, method = method,
                                     maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  p10 = numpy.zeros(_X.shape[1])
  p20 = numpy.e
  return(_beta0(_Y, _X))


#################### 02. Beta Regression: Varying Dispersion ####################


def _ll_betareg(y, x1, x2, beta1, beta2):
  """
  The function calculates the log likelihood function of a beta regression.
  Parameters:
    y     : the frequency outcome
    x1    : variables of the location parameter in beta regression
    x2    : variables of the dispersion parameter in beta regression
    beta1 : coefficients of the location parameter regression
    beta2 : coefficients of the dispersion parameter regression
  """

  mu = 1 / (1 + numpy.exp(-numpy.dot(x1, beta1)))
  ph = numpy.exp(numpy.dot(x2, beta2))
  _w = mu * ph
  _t = (1 - mu) * ph
  pr = scipy.special.gamma(_w + _t) * numpy.float_power(y, _w - 1) * numpy.float_power(1 - y, _t - 1) \
       / scipy.special.gamma(_w) / scipy.special.gamma(_t)
  ll = numpy.log(pr)
  return(ll)


def beta_reg(Y, X1, X2):
  """
  The function estimates a beta regression, assuming the varying dispersion.
  In the model output, coefficients starting with "MU:" are estimated for the 
  location parameter and coefficients starting with "PHI:" are estimated for 
  the dispersion parameter.
  Parameters:
    Y  : a pandas series for the fractional outcome in the range of (0, 1)
    X1 : a pandas dataframe with the location model variables that are all numeric values.
    X2 : a pandas dataframe with the dispersion model variables that are all numeric values.
  Example:
    beta_reg(Y, X1, X2).fit().summary() 
  """

  class _betareg(gll):
    def __init__(self, endog, exog, **kwds):
      super(_betareg, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      d1 = _X1.shape[1]
      beta1 = params[:d1]
      beta2 = params[d1:]
      ll = _ll_betareg(self.endog, self.exog[:, :d1], self.exog[:, d1:], beta1, beta2)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      if start_params == None:
        start_params = numpy.concatenate([p10, p20])
      return(super(_betareg, self).fit(start_params = start_params, method = method,
                                       maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X1 = X1.copy()
  _X2 = X2.copy()
  _X1.insert(loc = 0, column = "_CONST", value = 1)
  _X1.columns = ["MU:" + _ for _ in _X1.columns]
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["PHI:" + _ for _ in _X2.columns]
  _X = _X1.join(_X2)
  p10 = beta0_reg(Y, X1).fit(disp = 0).params[:-1]
  p20 = numpy.append(numpy.e, numpy.zeros(_X2.shape[1] - 1))
  return(_betareg(_Y, _X))


#################### 03. Simplex Regression: Fixed Dispersion ####################


def _ll_simplex0(y, x, beta, ln_s2):
  """
  The function calculates the log likelihood function of a simplex regression.
  Parameters:
    y     : the fractional outcome in the range of (0, 1)
    x     : variables of the location parameter in simplex regression
    beta  : coefficients of the location parameter regression
    ln_s2 : the dispersion parameter 
  """

  mu = 1 / (1 + numpy.exp(-numpy.dot(x, beta)))
  s2 = numpy.exp(ln_s2)
  _v = numpy.float_power(y, 3) * numpy.float_power(1 - y, 3)
  _d = numpy.float_power(y - mu, 2) / y / (1 - y) / numpy.float_power(mu, 2) \
       / numpy.float_power(1 - mu, 2)
  pr = numpy.float_power(2 * numpy.pi * s2 * _v, -0.5) * numpy.exp(-0.5 * _d / s2)
  ll = numpy.log(pr)
  return(ll)


def simplex0_reg(Y, X):
  """
  The function estimates a simplex regression, assuming a fixed dispersion.
  In the model output, coefficients are estimated for the location parameter.
  Parameters:
    Y : a pandas series for the fractional outcome in the range of (0, 1)
    X : a pandas dataframe with the location model variables that are all numeric values.
  Example:
    simplex0_reg(Y, X).fit().summary()   
  """

  class _simplex0(gll):
    def __init__(self, endog, exog, **kwds):
      super(_simplex0, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      beta = params[:-1]
      ln_s2  = params[-1]
      ll = _ll_simplex0(self.endog, self.exog, beta, ln_s2)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      self.exog_names.append('_LN_S2')
      if start_params == None:
        start_params = numpy.append(p10, p20)
      return(super(_simplex0, self).fit(start_params = start_params, method = method,
                                        maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X = X.copy()
  _X.insert(loc = 0, column = "_CONST", value = 1)
  p10 = numpy.append(1, numpy.zeros(_X.shape[1] - 1))
  p20 = numpy.e
  return(_simplex0(_Y, _X))


#################### 04. Simplex Regression: Varying Dispersion ####################


def _ll_simplexreg(y, x1, x2, beta1, beta2):
  """
  The function calculates the log likelihood function of a simplex regression.
  Parameters:
    y     : the fractional outcome in the range of (0, 1)
    x1    : variables of the location parameter in simplex regression
    x2    : variables of the dispersion parameter in simplex regression
    beta1 : coefficients of the location parameter regression
    beta2 : coefficients of the dispersion parameter regression
  """

  mu = 1 / (1 + numpy.exp(-numpy.dot(x1, beta1)))
  s2 = numpy.exp(numpy.dot(x2, beta2))
  _v = numpy.float_power(y, 3) * numpy.float_power(1 - y, 3)
  _d = numpy.float_power(y - mu, 2) / y / (1 - y) / numpy.float_power(mu, 2) \
       / numpy.float_power(1 - mu, 2)
  pr = numpy.float_power(2 * numpy.pi * s2 * _v, -0.5) * numpy.exp(-0.5 * _d / s2)
  ll = numpy.log(pr)
  return(ll)


def simplex_reg(Y, X1, X2):
  """
  The function estimates a simplex regression, assuming a varying dispersion.
  In the model output, coefficients starting with "MU:" are estimated for the 
  location parameter and coefficients starting with "S2:" are estimated for 
  the dispersion parameter.
  Parameters:
    Y  : a pandas series for the fractional outcome in the range of (0, 1)
    X1 : a pandas dataframe with the location model variables that are all numeric values.
    X2 : a pandas dataframe with the dispersion model variables that are all numeric values.
  Example:
    simplex_reg(Y, X1, X2).fit().summary()   
  """

  class _simplexreg(gll):
    def __init__(self, endog, exog, **kwds):
      super(_simplexreg, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
      d1 = _X1.shape[1]
      beta1 = params[:d1]
      beta2 = params[d1:]
      ll = _ll_simplexreg(self.endog, self.exog[:, :d1], self.exog[:, d1:], beta1, beta2)
      return(-ll)

    def fit(self, start_params = None, maxiter = 10000, maxfun = 5000, method = "ncg", **kwds):
      if start_params == None:
        start_params = numpy.append(p10, p20)
      return(super(_simplexreg, self).fit(start_params = start_params, method = method,
                                          maxiter = maxiter, maxfun = maxfun, **kwds))

  _Y = Y.copy()
  _X1 = X1.copy()
  _X2 = X2.copy()
  _X1.insert(loc = 0, column = "_CONST", value = 1)
  _X1.columns = ["MU:" + _ for _ in _X1.columns]
  _X2.insert(loc = 0, column = "_CONST", value = 1)
  _X2.columns = ["S2:" + _ for _ in _X2.columns]
  _X = _X1.join(_X2)
  p10 = simplex0_reg(Y, X1).fit(disp = 0).params[:-1]
  p20 = numpy.append(numpy.e, numpy.zeros(_X2.shape[1] - 1))
  return(_simplexreg(_Y, _X))


