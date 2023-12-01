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
"""Plotting functions for the transformer model."""
import functools
import pickle

import flax.linen as nn
from causallm_icl import lr_algos
from causallm_icl import sampler_lib
from causallm_icl import utils
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.io import gfile

Array = utils.Array


def binclass_error(y1: Array, y2: Array) -> Array:
  return 1.0 * (y1 * y2 <= 0).mean(axis=(0, 2))


def algo_fns_def(xdim: int):
  return {
      "kernel":
          functools.partial(lr_algos.KRAlgorithm),
      "kde": functools.partial(lr_algos.KDEAlgorithm),
      "Ridge(0.1)":
          functools.partial(lr_algos.RidgeRegressionAlgorithm, alpha=0.1),
      "SGD(0.1, w=full, iter=10)":
          functools.partial(
              lr_algos.SGD,
              xdim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.1,
              weight_decay=0.0,
              window=-1,
              gd_iter=10
          ),
      "SGD(0.1, w=full, iter=20)":
          functools.partial(
              lr_algos.SGD,
              xdim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.1,
              weight_decay=0.0,
              window=-1,
              gd_iter=20
          ),
      "SGD(0.1, w=full, iter=5)":
          functools.partial(
              lr_algos.SGD,
              xdim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.1,
              weight_decay=0.0,
              window=-1,
              gd_iter=5
          ),
      "SGD(0.1, w=full, iter=15)":
          functools.partial(
              lr_algos.SGD,
              xdim,
              nn.initializers.zeros,
              learning_rate_fn=lambda i: 0.1,
              weight_decay=0.0,
              window=-1,
              gd_iter=15
          ),
  }


def plot_empirical_distribution(
    model,
    params,
    seq: Array,
    coefficients: Array,
    xs: Array,
    ys: Array,
    ys_true: Array,
    prefix: str,
    save_predictions: bool = True,
    plot_step=1,
    start_step=1,
    output_map="linear",
    use_enc_mask=False,
):
  """Calculates the empirical weights for the model's predictive function.

  Args:
    model (predictor_flax.CausalLM): causal transformer trained for icl.
    params (Any): model params.
    seq (torch.Tensor): input tensor with shape (batch_size, seq_length,
      hidden_size)
    coefficients (Optional[torch.Tensor], optional): Gold weights. Defaults to
      None.
    xs: x sequence.
    ys: y sequence.
    ys_true: the groundtruth y sequence.
    prefix (str): save prefix.
    save_predictions (bool): whether to save the predictions
    plot_step: plot every few steps
    start_step: starting index of the example in the sequence for plotting. Do
      not plot the examples before start_step but use them as context.
    output_map: a (non-)linear output mapping to study non-linear functions.
    use_enc_mask: use enc_mask for the first few examples

  Returns:
    torch.Tensor: empirical weights with shape (x_dim).
  """
  # algo_fns = algo_fns_def(xdim=seq.shape[-1]-1)
  predictions_dict = {}
  del coefficients, xs, ys

  y_hidden, attn = model.apply(
      {"params": params}, inputs=seq, train=False, return_attention=True,
      hidden_loss=False, use_enc_mask=use_enc_mask, start_step=start_step)

  predictions_dict["Gold"] = np.array(ys_true)
  predictions_dict["attn"] = np.array(attn)

  ax = plt.axes()
  for i, yhid in enumerate(y_hidden):
    hid_errors = ((yhid - ys_true)**2).mean(axis=(0, 2))
    # hid_errors = binclass_error(yhid, ys_true)
    hid_errors_plot = hid_errors[start_step::plot_step]
    ax.plot(np.arange(
        start_step, len(hid_errors_plot) * plot_step + start_step, plot_step),
            hid_errors_plot, label=f"hid{i}")
    predictions_dict[f"hid{i}"] = np.array(yhid)

  ax.legend()
  fpath = prefix + "errors.jpeg"
  plt.title("Mean Prediction Loss after #Exemplars")
  ax.set_yscale("log")
  ax.set_xlabel("#Exemplars")
  ax.set_ylabel("MSE per unit")
  with gfile.GFile(fpath, "wb") as handle:
    plt.savefig(handle, dpi=300)
  plt.close()

  # plot the attention
  for layer in range(predictions_dict["attn"].shape[0]):
    ax = plt.axes()
    plt.imshow(predictions_dict["attn"][layer].mean(axis=(0, 1)),
               cmap="hot", interpolation="nearest")
    fpath = prefix + f"attn{layer}.jpeg"
    with gfile.GFile(fpath, "wb") as handle:
      plt.savefig(handle, dpi=300)
      plt.close()

  y2x = sampler_lib.inv_map_func(output_map)(ys_true)
  for algo_name, pred_algo in predictions_dict.items():
    if (algo_name == "Gold" or algo_name == "attn"
        or algo_name == "werr" or algo_name == "trh"): continue
    ax = plt.axes()
    if algo_name.startswith("hid"):
      for i in range(ys_true.shape[0]):
        plt.scatter(y2x[i, start_step::plot_step, 0],
                    pred_algo[i, start_step::plot_step, 0])
    elif algo_name.startswith("y2hid"):
      for i in range(ys_true.shape[0]):
        plt.scatter(y2x[i, start_step::plot_step, 0],
                    pred_algo[i, start_step::plot_step, 0])
    else:
      for i in range(ys_true.shape[0]):
        plt.scatter(y2x[i, start_step::plot_step, 0],
                    pred_algo[i, :, 0])

    fpath = prefix + f"scatter_{algo_name}.jpeg"
    with gfile.GFile(fpath, "wb") as handle:
      plt.savefig(handle, dpi=300)
      plt.close()

  if save_predictions:
    fpath = prefix + "predictions.pkl"
    with gfile.GFile(fpath, "wb") as handle:
      pickle.dump(predictions_dict, handle)


def plot_empirical_distribution_mc(
    model,
    params,
    seq: Array,
    ys_true: Array,
    prefix: str,
    save_predictions: bool = True,
    plot_step=1,
    start_step=1,
    use_enc_mask=False,
):
  """Calculates the empirical weights for the model's predictive function.

  Args:
    model (predictor_flax.CausalLM): causal transformer trained for icl.
    params (Any): model params.
    seq (torch.Tensor): input tensor with shape (batch_size, seq_length,
      hidden_size)
    ys_true: the groundtruth y sequence.
    prefix (str): save prefix.
    save_predictions (bool): whether to save the predictions
    plot_step: plot every few steps
    start_step: starting index of the example in the sequence for plotting. Do
      not plot the examples before start_step but use them as context.
    use_enc_mask: use enc_mask for the first few examples

  Returns:
    torch.Tensor: empirical weights with shape (x_dim).
  """
  predictions_dict = {}

  y_hidden, attn = model.apply(
      {"params": params}, inputs=seq, train=False, return_attention=True,
      hidden_loss=False, use_enc_mask=use_enc_mask, start_step=start_step)

  predictions_dict["Gold"] = np.array(ys_true)
  predictions_dict["attn"] = np.array(attn)

  ax = plt.axes()
  for i, yhid in enumerate(y_hidden):
    hid_errors = (1.0 * (np.argmax(yhid, axis=-1) ==
                         np.argmax(ys_true, axis=-1))).mean(axis=0)
    hid_errors_plot = hid_errors[start_step::plot_step]
    ax.plot(np.arange(
        start_step, len(hid_errors_plot) * plot_step + start_step, plot_step),
            hid_errors_plot, label=f"hid{i}")
    predictions_dict[f"hid{i}"] = np.array(yhid)

  ax.legend()
  fpath = prefix + "errors.jpeg"
  plt.title("Mean Prediction Loss after #Exemplars")
  ax.set_xlabel("#Exemplars")
  ax.set_ylabel("MSE per unit")
  with gfile.GFile(fpath, "wb") as handle:
    plt.savefig(handle, dpi=300)
  plt.close()

  if save_predictions:
    fpath = prefix + "predictions.pkl"
    with gfile.GFile(fpath, "wb") as handle:
      pickle.dump(predictions_dict, handle)
