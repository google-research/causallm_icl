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
"""Sampler library for data generation."""
from typing import Callable, Tuple

import numpy as np


def str_to_distribution_fn(distribution_str: str) -> Callable[..., np.ndarray]:
  """Convert string representation to function."""
  mixtures = distribution_str.split(",")
  if len(mixtures) > 1:
    fns = [str_to_distribution_fn(mixture) for mixture in mixtures]

    def mixture_sample_fn(*args, **kwargs):
      samples = [fn(*args, **kwargs) for fn in fns]
      samples = np.stack(samples, axis=0)
      flat = samples.reshape(samples.shape[0], -1)
      indexer = np.random.randint(flat.shape[0], size=flat.shape[-1])
      flat = flat[indexer, np.arange(flat.shape[-1])]
      return flat.reshape(*samples.shape[1:])

    return mixture_sample_fn
  else:
    distribution_type, beta = distribution_str.split("+")
    distribution_type, alpha = distribution_type.split("*")
    alpha = float(alpha)
    beta = float(beta)
    if distribution_type == "uniform":
      distribution_fn = np.random.rand
    elif distribution_type == "normal":
      distribution_fn = np.random.randn
    else:
      raise ValueError("Unknown distribution type.")

    def distribution_fn_scaled(*args, **kwargs):
      return distribution_fn(*args, **kwargs) * alpha + beta

    return distribution_fn_scaled


def str_to_gmm_mix_fn(x_distribution_str,
                      w_distribution_str) -> Callable[..., np.ndarray]:
  """Convert string representation to mixture sampling function."""
  x_mixtures = x_distribution_str.split(",")
  x_fns = [str_to_distribution_fn(mixture) for mixture in x_mixtures]

  w_mixtures = w_distribution_str.split(",")
  w_fns = [str_to_distribution_fn(mixture) for mixture in w_mixtures]

  assert len(x_fns) == len(w_fns)
  nmix = len(x_fns)

  def mixture_sample_fn(n, length, dim, num_classes=1):
    x_samples = []
    w_samples = []
    classes = np.random.randint(nmix, size=n)
    for c in classes:
      x_samples.append(x_fns[c](length, dim))
      if num_classes <= 1:
        w_samples.append(w_fns[c](dim, 1))
      else:
        w_samples.append(w_fns[c](dim, num_classes))
    return np.stack(x_samples, axis=0), np.stack(w_samples, axis=0)

  return mixture_sample_fn


def map_func(name: str):
  if name == "linear":
    return (lambda y: y)
  elif name == "sigmoid":
    return (lambda y: 1.0 / (1.0 + np.exp(-y * 1)) - 0.5)
  elif name == "0-1":
    return (lambda y: 2.0 * (y >= 0) - 1.)


def inv_map_func(name: str):
  if name == "linear":
    return (lambda y: y)
  elif name == "sigmoid":
    return (lambda y: np.log((y + 0.5) / (0.5 - y)) / 1.0)
  elif name == "0-1":
    return (lambda y: y)


class LRSampler(object):
  """Samples linear regression data from specified distributions."""

  def __init__(
      self,
      length: int,
      dim: int,
      num_dec_examples: int = 20,
      cxw_gmm_fn=None,
      noise_std: float = 0.0,
      final_xdim_one=False,
      output_map="linear"
  ):
    """Initializes the sampler.

    Args:
      length (int): Number of examplers to generate.
      dim (int): dimension of the x vectors.
      num_dec_examples: number of query examples.
      cxw_gmm_fn: mixture of clusters of x and w functions
      noise_std (float): adds gaussian noise if the value > 0.0. Default is 0.0
      final_xdim_one: last x dim is always one, used as offset.
      output_map: nonlinear output map
    """
    self.length = length
    self.num_dec_examples = num_dec_examples
    self.num_enc_examples = length - num_dec_examples
    self.dim = dim

    self.cxw_gmm_fn = cxw_gmm_fn
    self.noise_std = noise_std
    self.final_xdim_one = final_xdim_one
    self.output_map = output_map

  def sample(
      self, n: int = 1, scale: float = 1.0) -> Tuple[np.ndarray, ...]:
    """Generates a random sequence of x and y vector comes from a linear regressor.

    Args:
      n (int, optional): batch size. Defaults to 1.
      scale: scale the input x.

    Returns:
      Tuple[np.array, np.array]: x,y sequences, weights of the regressor
    """
    x, w = self.cxw_gmm_fn(n, self.length, self.dim)
    x_scaled = x * scale
    x = np.concatenate([x[:, :self.dim + 1, :],
                        x_scaled[:, self.dim + 1:, :]], axis=1)

    if self.final_xdim_one:
      x[:, :, -1] = 1.

    x_vec = np.concatenate([np.zeros((n, self.length, 1)), x], axis=-1)

    y = np.einsum("bli,bic->blc", x, w)
    y = map_func(self.output_map)(y)
    if self.noise_std > 0:
      y += self.noise_std * np.random.randn(*y.shape)

    y_vec = np.concatenate([y, x], axis=-1)

    # out = np.stack([x_vec, y_vec], axis=2)
    # out = np.reshape(out, [n, self.length * 2, self.dim + 1])
    out = np.concatenate([y_vec[:, :self.num_enc_examples, :],
                          x_vec[:, self.num_enc_examples:, :]], axis=1)
    return out, w, x, y


class MCSampler:
  """Samples multiclass classification data from specified distributions."""

  def __init__(
      self,
      length: int,
      dim: int,
      num_classes: int,
      num_dec_examples: int = 20,
      cxw_gmm_fn=None,
      noise_std: float = 0.0,
      final_xdim_one=False,
  ):
    """Initializes the sampler.

    Args:
      length (int): Number of examplers to generate.
      dim (int): dimension of the x vectors.
      num_classes: number of classes.
      num_dec_examples: number of query examples.
      cxw_gmm_fn: mixture of clusters of x and w functions
      noise_std (float): adds gaussian noise if the value > 0.0. Default is 0.0
      final_xdim_one: last x dim is always one, used as offset.
    """
    self.length = length
    self.dim = dim
    self.num_classes = num_classes
    self.num_dec_examples = num_dec_examples
    self.num_enc_examples = length - num_dec_examples
    self.cxw_gmm_fn = cxw_gmm_fn
    self.noise_std = noise_std
    self.final_xdim_one = final_xdim_one

  def sample(
      self, n: int = 1, scale: float = 1.0) -> Tuple[np.ndarray, ...]:
    """Generates a random sequence of x and y vector comes from a linear regressor.

    Args:
      n (int, optional): batch size. Defaults to 1.
      scale: scale the input x.

    Returns:
      Tuple[np.array, np.array]: x,y sequences, weights of the regressor
    """
    x, w = self.cxw_gmm_fn(n, self.length, self.dim, self.num_classes)
    x_scaled = x * scale
    x = np.concatenate([x[:, :self.dim + 1, :],
                        x_scaled[:, self.dim + 1:, :]], axis=1)

    if self.final_xdim_one:
      x[:, :, -1] = 1.

    x_vec = np.concatenate(
        # [-np.ones((n, self.length, self.num_classes)) / self.num_classes, x],
        [np.zeros((n, self.length, self.num_classes)), x],
        axis=-1)

    logit = np.einsum("bli,bic->blc", x, w)
    if self.noise_std > 0:
      logit += self.noise_std * np.random.randn(*logit.shape)

    # y = logit
    y = 1.0 * (logit == np.max(logit, axis=-1, keepdims=True))
    # logit *= 6.
    # y = np.exp(logit) / np.sum(np.exp(logit), axis=-1, keepdims=True)
    y_vec = np.concatenate(
        # [y - 1. / self.num_classes, x],
        [y, x],
        axis=-1)

    # out = np.stack([x_vec, y_vec], axis=2)
    # out = np.reshape(out, [n, self.length * 2, self.dim + self.num_classes])
    out = np.concatenate([y_vec[:, :self.num_enc_examples, :],
                          x_vec[:, self.num_enc_examples:, :]], axis=1)
    return out, w, x, y
