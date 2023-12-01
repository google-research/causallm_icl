# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides algorithms for linear regression problems."""
import abc
from typing import Any, Callable, List, Mapping, Optional, Tuple
import warnings

import flax.linen as nn
from flax.training import common_utils
from causallm_icl import utils
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import exceptions as sklearn_exceptions
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors

warnings.filterwarnings(
    "ignore", category=sklearn_exceptions.UndefinedMetricWarning)


class RegressionAlgorithm(metaclass=abc.ABCMeta):
  """A regression learning algorithm that learns predictor."""

  @abc.abstractmethod
  def fit(self, x: utils.Array, y: utils.Array):
    """Fits regression to a data given inputs x and outputs y.

    Args:
      x (utils.Array): Inputs w. shape (n_samples, input_features).
      y (utils.Array): Outputs w. shape (n_samples, output_features).
    """

  @abc.abstractmethod
  def predict(self, x: utils.Array) -> utils.Array:
    """Produce predictions for new inputs .

    Args:
      x (utils.Array): Inputs w. shape (n_samples, input_features).

    Returns:
      utils.Array: Predictions w. shape (n_samples, output_features).
    """

  def get_parameters(self) -> Optional[Mapping[str, utils.Array]]:
    """Returns the parameters of the algorithm.

    Returns:
      utils.Array: model parameters.
    """
    return None

  def iterate(self, x: utils.Array, y: utils.Array):
    """Iterate the algorithm over the data given inputs x and outputs y."""
    raise NotImplementedError("This algorithm does not support iterate (yet)")

  def is_iterative(self):
    """Returns whether the algorithm is iterative.

    Returns:
      bool: True if the algorithm is iterative.
    """
    return False

  @abc.abstractmethod
  def reset(self):
    """Resets the algorithm."""

  def scores(self,
             y: utils.Array,
             y_hat: Optional[utils.Array] = None) -> Mapping[str, utils.Array]:
    """Gets fit statistics such as R2 and MSE.

    Args:
      y (utils.Array): gold outputs.
      y_hat (utils.Array): predictions.

    Returns:
      Mapping[str, float]: fit statistics.
    """
    if y.ndim == 3:
      r2_scores = [
          metrics.r2_score(y[:, i, :], y_hat[:, i, :])
          for i in range(y.shape[1])
      ]
      r2_value = np.array(r2_scores)
      mse_value = ((y_hat - y)**2).mean(axis=(0, -1))
    else:
      r2_value = metrics.r2_score(y, y_hat)
      mse_value = ((y_hat - y)**2).mean(axis=-1)

    return {"R2": r2_value, "MSE": mse_value}


class LeastSquareAlgorithm(RegressionAlgorithm):
  """Least square regression algorithm."""

  def __init__(self, seed: int = 0, fit_intercept: bool = False) -> None:
    self.seed = seed
    self.fit_intercept = fit_intercept
    self.is_fitted = False
    self.regressor = linear_model.LinearRegression(
        fit_intercept=self.fit_intercept)

  def reset(self):
    self.regressor = None
    self.is_fitted = False

  def fit(self, x: utils.Array, y: utils.Array):
    assert x.ndim == 2 and y.ndim == 2
    self.regressor.fit(x, y)
    self.is_fitted = True

  def predict(self, x: utils.Array) -> utils.Array:
    assert x.ndim == 2
    return self.regressor.predict(x)

  def get_parameters(self) -> Mapping[str, utils.Array]:
    return {"W": self.regressor.coef_, "b": self.regressor.intercept_}


class RidgeRegressionAlgorithm(RegressionAlgorithm):
  """Ridge regression algorithm with regularized least square."""

  def __init__(self,
               alpha: float = 0.01,
               fit_intercept: bool = False,
               seed: int = 0) -> None:
    self.alpha = alpha
    self.seed = seed
    self.fit_intercept = fit_intercept
    self.is_fitted = False
    self.regressor = linear_model.Ridge(
        alpha=self.alpha,
        fit_intercept=self.fit_intercept,
        random_state=self.seed)

  def reset(self):
    self.regressor = None
    self.is_fitted = False

  def fit(self, x: utils.Array, y: utils.Array):
    assert x.ndim == 2 and y.ndim == 2
    self.regressor.fit(x, y)
    self.is_fitted = True

  def predict(self, x: utils.Array) -> utils.Array:
    assert x.ndim == 2
    return self.regressor.predict(x)

  def get_parameters(self) -> Mapping[str, utils.Array]:
    return {"W": self.regressor.coef_, "b": self.regressor.intercept_}


class KNNAlgorithm(RegressionAlgorithm):
  """KNN regression algorithm."""

  def __init__(self,
               k: int = 5,
               weighting: str = "uniform",
               seed: int = 0) -> None:
    self.k = k
    self.seed = seed
    self.weighting = weighting
    self.is_fitted = False
    self.regressor = neighbors.KNeighborsRegressor(k, weights=self.weighting)

  def reset(self):
    self.regressor = None
    self.is_fitted = False

  def fit(self, x: utils.Array, y: utils.Array):
    assert x.ndim == 2 and y.ndim == 2
    k = min(self.k, x.shape[0])
    self.regressor = neighbors.KNeighborsRegressor(k, weights=self.weighting)
    self.regressor.fit(x, y)
    self.is_fitted = True

  def predict(self, x: utils.Array) -> utils.Array:
    assert x.ndim == 2
    return self.regressor.predict(x)


class SGD(RegressionAlgorithm):
  """Stochastic gradient descent variants."""

  def __init__(
      self,
      dim: int,
      init_fn: Callable[..., utils.Array],
      learning_rate_fn: Callable[[int], float],
      weight_decay: float = 0.0,
      window: int = 1,
      seed: int = 0,
      gd_iter: int = 1
  ) -> None:
    self.learning_rate_fn = learning_rate_fn
    self.window = window
    self.init_fn = init_fn
    self.x_dim = dim
    self.seed = seed
    self.key = jax.random.PRNGKey(self.seed)
    self.weight = None
    self.is_fitted = False
    self.weight_decay = weight_decay
    self.gd_iter = gd_iter

  def init_weight(self):
    self.weight = self.init_fn(self.key, (self.x_dim, 1))

  def reset(self):
    self.init_weight()
    self.is_fitted = False

  def iterate(self, x: utils.Array, y: utils.Array):
    assert x.ndim == 2 and y.ndim == 2

    if self.weight is None:
      self.init_weight()

    for i in range(x.shape[0]):
      if self.window == -1:
        start = 0
      else:
        start = min(0, i - self.window)

      x_batch = x[start:i + 1]
      grad = -2 * x_batch.T @ (y[start:i + 1] - self.predict(x_batch))
      if self.weight_decay > 0:
        grad += 2 * self.weight_decay * self.weight

      # grad = jax.lax.clamp(-20.0, grad, 20.0)
      self.weight -= self.learning_rate_fn(i) * grad  # / x_batch.shape[0]

  def gd(self, x: utils.Array, y: utils.Array, print_error=False):
    """Gradient descent."""
    assert x.ndim == 2 and y.ndim == 2
    xdim = x.shape[1]

    if self.weight is None:
      self.init_weight()

    error_list = []
    for i in range(self.gd_iter):
      y_wx = y - self.predict(x)
      grad = -2 * x.T @ y_wx / x.shape[0]
      grad *= 1. / np.linalg.norm(grad) * np.sqrt(float(xdim))
      if self.weight_decay > 0:
        grad += 2 * self.weight_decay * self.weight

      self.weight -= self.learning_rate_fn(i) * grad
      error_list.append(np.linalg.norm(y_wx) ** 2)
    if print_error:
      print(error_list)

  def fit(self, x: utils.Array, y: utils.Array):
    assert x.ndim == 2 and y.ndim == 2
    self.init_weight()
    if self.window == -1:
      self.gd(x, y)
    else:
      self.iterate(x, y)
    self.is_fitted = True

  def predict(self, x: utils.Array) -> utils.Array:
    assert x.ndim == 2
    return x @ self.weight

  def is_iterative(self):
    return self.window == 1


class MeanAlgorithm(RegressionAlgorithm):
  """Mean of y."""

  def __init__(self) -> None:
    self.mean = None
    self.is_fitted = False

  def reset(self):
    self.mean = None
    self.is_fitted = False

  def fit(self, x: utils.Array, y: utils.Array):
    self.mean = jnp.mean(y, axis=0, keepdims=True)

  def predict(self, x: utils.Array):
    return self.mean


class KRAlgorithm(RegressionAlgorithm):
  """kernel regression regression algorithm."""

  def __init__(self,
               beta=0.01) -> None:
    self.beta = beta
    self.is_fitted = False

  def reset(self):
    self.is_fitted = False

  def fit(self, x: utils.Array, y: utils.Array):
    assert x.ndim == 2 and y.ndim == 2
    xdim = x.shape[1]
    self.x = x
    self.b = jnp.mean(y, axis=0, keepdims=True)  # center y
    n = x.shape[0]
    exx = jnp.exp(
        -0.5 / xdim * jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    exx += self.beta * jnp.eye(n)
    self.v = jnp.linalg.solve(exx, y - self.b)

  def predict(self, x: utils.Array) -> utils.Array:
    assert x.ndim == 2
    xdim = x.shape[1]
    extx = jnp.exp(
        -0.5 / xdim *
        jnp.sum((x[:, None, :] - self.x[None, :, :]) ** 2, axis=-1))
    ycme = jnp.matmul(extx, self.v) + self.b
    return ycme


class KDEAlgorithm(RegressionAlgorithm):
  """kernel density estimation algorithm."""

  def __init__(self) -> None:
    self.is_fitted = False

  def reset(self):
    self.is_fitted = False

  def fit(self, x: utils.Array, y: utils.Array):
    assert x.ndim == 2 and y.ndim == 2
    self.x = x
    self.y = y

  def predict(self, x: utils.Array) -> utils.Array:
    assert x.ndim == 2
    xdim = x.shape[1]
    kd = nn.softmax(
        -0.5 / xdim * jnp.sum((x[:, None, :] - self.x[None, :, :]) ** 2,
                              axis=-1))  # [1, L]
    y = jnp.matmul(kd, self.y)
    return y


def online_regression(
    algo_fn: Callable[..., RegressionAlgorithm],
    x: utils.Array,
    y: utils.Array,
    xt: utils.Array,
    step: int = 1,
    start_step: int = 1
) -> Tuple[List[Any], ...]:
  """Runs online regression for linear algorithms."""

  assert x.ndim == 2 and y.ndim == 2
  predictions, parameters = [], []
  algo = algo_fn()
  for i in range(start_step, xt.shape[0], step):
    if algo.is_iterative():
      algo.iterate(x[i - 1:i, :], y[i - 1:i, :])
    else:
      algo = algo_fn()
      algo.fit(x[:i, :], y[:i, :])
    xt_i = xt[i:i + 1, :]
    y_hat = algo.predict(xt_i).squeeze(axis=0)
    predictions.append(y_hat)
    parameters.append(algo.get_parameters())
  return tuple(map(common_utils.stack_forest, (predictions, parameters)))


def online_regression_with_batch(
    algo_fn: Callable[..., RegressionAlgorithm],
    xs: utils.Array,
    ys: utils.Array,
    ys_true: utils.Array,
    xt=None,
    step=1,
    start_step=1
):
  """Runs online regression for linear algorithms for a batch."""
  batched_predictions, batched_parameters = [], []
  for i in range(xs.shape[0]):
    x_batch = xs[i, ...]
    y_batch = ys[i, ...]
    if xt is None:
      xt_batch = x_batch
    else:
      xt_batch = xt[i, ...]
    predictions, parameters = online_regression(
        algo_fn, x_batch, y_batch, xt_batch, step, start_step)
    batched_predictions.append(predictions)
    batched_parameters.append(parameters)

  batched_predictions, batched_parameters = tuple(
      map(common_utils.stack_forest, (batched_predictions, batched_parameters)))
  batched_errors = algo_fn().scores(ys_true[:, start_step::step, :],
                                    batched_predictions)
  return batched_predictions, batched_parameters, batched_errors
