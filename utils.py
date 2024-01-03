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
"""Utils for incontext library."""

import random
from typing import Any, Callable, List, Union
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]
Dtype = Any


class ConfigDict(object):
  """Simple config dict."""

  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, my_dict):
    self.initial_dict = my_dict
    for key in my_dict:
      setattr(self, key, my_dict[key])

  def __str__(self) -> str:
    name = ""
    for k, v in self.initial_dict.items():
      name += str(k) + "=>" + str(v) + "\n"
    return name


def set_seed(seed: int) -> List[int]:
  """Sets the seed for the random number generator.

  Args:
    seed (int): seed for the random number generator.

  Returns:
    List[int]: list of seeds used.
  """
  random.seed(seed)
  np.random.seed(seed)
  return [seed]


def flags_to_args():
  # pylint: disable=protected-access
  flags_dict = {k: v.value for k, v in flags.FLAGS.__flags.items()}
  # pylint: enable=protected-access
  return ConfigDict(flags_dict)


def y_x_fn(x: Array, hid: int, seq: Array, start_step: int, end_step: int,
           model: Any, params: Any, use_enc_mask: bool = False,
           num_classes: int = 1) -> Any:
  """Output y as a function of x for any model with params.

  Args:
    x: input array for differentiation, it is part of seq
    hid: the hidden layer id as output y
    seq: the reference array for building the complete input sequence.
    start_step: the start step of x in seq
    end_step: the end step of x in seq
    model: the NN model
    params: the params of the model
    use_enc_mask: apply the enc masks on in-context examples.
    num_classes: number of classes, default 1 for linear regression

  Returns:
    a scalar y as the sum of outputs across batches and positions.
  """
  # First few examples [0x1, y1x1]
  seq1 = seq[:, :start_step * 2, :]

  # from [x2, x3] to [0x2, 0x3]
  seq2 = jnp.concatenate([jnp.zeros([x.shape[0], x.shape[1], 1]), x], axis=-1)
  # from [0x2, 0x3] to [0x2, y2x2, 0x3, y3x3]
  seq2 = jnp.stack(
      [seq2, seq[:, np.arange(start_step, end_step) * 2 + 1, :]], axis=2)
  seq2 = jnp.reshape(seq2, [seq2.shape[0], -1, seq2.shape[-1]])

  # concatenate to get [0x1, y1x1, 0x2, y2x2, 0x3, y3x3]
  seq1 = jnp.concatenate([seq1, seq2], axis=1)
  y_hid = model.apply(
      {"params": params}, inputs=seq1, train=False, return_y=hid,
      use_enc_mask=use_enc_mask, start_step=start_step)
  return jnp.sum(y_hid[:, np.arange(start_step, end_step), :num_classes])


def trh(loss_fn: Callable[..., Any], x: Array, hid: int,
        nsamples: int = 10) -> Any:
  """Estimate the trH of d^2 y/ dx^2 of loss_fn using the Huchinson method.

  Args:
    loss_fn: the loss function
    x: the input
    hid: the hidden layer id as output y
    nsamples: number of samples for Huchinson method.

  Returns:
    a scalar as an estimate of the trace of Hessian.
  """
  def _grad_sum(x_, rv_):
    g = jax.grad(loss_fn)(x_, hid)
    return jnp.sum(g * rv_)

  sum_trh = 0.
  for _ in range(nsamples):
    rv = np.random.normal(size=x.shape)
    h = jax.grad(_grad_sum)(x, rv)
    sum_trh += jnp.sum(h * rv, axis=-1)
  return sum_trh / nsamples
