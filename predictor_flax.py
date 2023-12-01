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
"""Transformer predictor."""
from typing import Any, Optional, Union

from absl import flags
import flax.linen as nn
from causallm_icl import transformer_lib_flax
import jax
import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]

flags.DEFINE_bool("loss_on_x_steps", default=False, help="Take loss on x steps")


def extract_y(seq: Array, offset: int = 0, num_classes: int = 1) -> Array:
  """Extracts the y vector from the input tensor.

  Args:
          seq (torch.Tensor): tensor with shape (batch_size, seq_length,
            hidden_size)
          offset (int, optional): optional offset for where ys start. Defaults
            to 0.
          num_classes (int, optional): number of classes. default = 1.

  Returns:
          torch.Tensor: tensor with shape (batch_size, num_exemplars,
          hidden_size)
  """
  return seq[:, jnp.arange(offset, seq.shape[1], 1), :num_classes]


def l2_loss(y1: Array, y2: Any = 0.) -> Array:
  return ((y1 - y2)**2).sum(axis=-1)


def logistic_loss(y1: Array, y2: Optional[Array] = None) -> Array:
  h2 = 0.
  if y2 is None:
    y2 = 0.5
    h2 = -np.log(2.)
  else:
    y2 = (y2 + 1) / 2.
  return h2 + (y2 * jnp.log(1. + jnp.exp(-y1)) +
               (1 - y2) * jnp.log(1. + jnp.exp(y1))).sum(axis=-1)


def ce_loss(y1: Array, y2: Optional[Array] = None) -> Array:
  num_class = y1.shape[-1]
  h2 = 0.
  if y2 is None:
    y2 = np.ones(y1.shape) / num_class
    h2 = (y2 * np.log(y2)).sum(axis=-1)
  return h2 - (y2 * nn.log_softmax(y1, axis=-1)).sum(axis=-1)


class SimpleNN(nn.Module):
  """A simple one layer NN."""
  config: transformer_lib_flax.TransformerConfig
  @nn.compact
  def __call__(
      self,
      inputs: Array,
      labels: Optional[Array] = None,
      train: Optional[bool] = False,
      return_attention: Optional[bool] = False,
      hidden_loss: Optional[bool] = False,
      return_y: int = -1
  ):
    """CausalLM is a transformer based auto-regressive model.

    It predicts the next continuous vector based on the previous
    vectors.
    Args:
      inputs (Array): input tensor
      labels: Array,
      train (bool): training mode
      return_attention (bool): whether to return attentions
      hidden_loss: add the loss on hidden layers.
      return_y: if >=0, only return the y of the layer.

    Returns:
      Tuple[Array, Tuple[Array, ...]]: Tuple of loss and extras
    """
    config = self.config
    seq_from = inputs[:, :-1, :]

    y_hidden = []
    hidden_pred = nn.Dense(
        1,
        kernel_init=config.linear_w_init,
        bias_init=config.linear_bias_init)(seq_from)
    y_hidden.append(extract_y(hidden_pred, offset=0))
    y_pred = y_hidden[-1]
    seq_hiddens = y_hidden
    attn_weights = np.zeros([1, 1, 1, 1, 1])

    if return_y >= 0:
      return y_hidden[return_y]

    y_target = labels
    y_errors = ((y_pred - y_target)**2).sum(axis=-1)
    errors = y_errors

    if hidden_loss:
      for yhid in y_hidden:
        errors += ((yhid - y_target)**2).sum(axis=-1)

    if return_attention:
      return errors, (y_errors, y_pred, y_hidden, seq_hiddens, attn_weights)
    else:
      return errors, (y_errors, y_pred, y_hidden, seq_hiddens)

  def extract_y(self, seq: Array, offset: int = 0) -> Array:
    return extract_y(seq, offset=offset)


class CausalLM(SimpleNN):
  """CausalLM model that predicts next vector or only the next y."""
  config: transformer_lib_flax.TransformerConfig

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      labels: Optional[Array] = None,
      train: Optional[bool] = False,
      return_attention: Optional[bool] = False,
      hidden_loss: Optional[bool] = False,
      return_y: int = -1,
      use_enc_mask: Optional[bool] = False,
      start_step: Optional[int] = 20,
  ):
    """CausalLM is a transformer based auto-regressive model.

    It predicts the next continuous vector based on the previous
    vectors.
    Args:
      inputs (Array): input tensor
      labels (Array): label
      train (bool): training mode
      return_attention (bool): whether to return attentions
      hidden_loss: add the loss on hidden layers.
      return_y: if >=0, only return the y of the layer.
      use_enc_mask: apply the enc masks on in-context examples.
      start_step: start step for decoding

    Returns:
      Tuple[Array, Tuple[Array, ...]]: Tuple of loss and extras
    """
    config = self.config
    seq_from = inputs  # [:, :-1, :]

    # apply mask of shape [batch, head=1, L, L]
    batch = seq_from.shape[0]
    seq_len = seq_from.shape[1]
    assert start_step < seq_len

    mask = np.concatenate(
        [np.ones([batch, 1, seq_len, start_step]),
         np.zeros([batch, 1, seq_len, seq_len - start_step])],
        axis=-1)
    if not use_enc_mask:
      mask *= np.tril(np.ones([batch, 1, seq_len, seq_len]))

    _, seq_hiddens, attn_weights = transformer_lib_flax.Transformer(
        config)(
            inputs=seq_from,
            mask=mask,
            train=train,
            return_attention=return_attention)

    num_classes = config.num_classes
    if config.loss_on_x_steps:
      output_shape = seq_from.shape[-1]
    else:
      output_shape = num_classes

    y_hidden = []
    for i, hidden in enumerate(seq_hiddens):
      if i == 0:
        continue  # skip the 0th layer which is just the embedding.
      if i != len(seq_hiddens) - 1 and train:
        hidden = jax.lax.stop_gradient(hidden)
      hidden_pred = nn.Dense(
          output_shape,
          kernel_init=config.linear_w_init,
          bias_init=config.linear_bias_init)(hidden)  # probe for each layer
      y_hidden.append(extract_y(hidden_pred, offset=0,
                                num_classes=num_classes))
    y_pred = y_hidden[-1]

    if return_y >= 0:
      return y_hidden[return_y]
    if labels is None:
      if return_attention:
        return y_hidden, attn_weights
      else:
        return y_hidden

    y_target = jnp.concatenate([
        np.zeros([batch, start_step, num_classes]),
        labels[:, start_step:, :]], axis=1)

    y_errors = l2_loss(y_pred, y_target)
    errors = y_errors

    if hidden_loss:
      for yhid in y_hidden:
        errors += l2_loss(yhid, y_target)

    if return_attention:
      return errors, (y_errors, y_pred, y_hidden, attn_weights)
    else:
      return errors, (y_errors, y_pred, y_hidden)
