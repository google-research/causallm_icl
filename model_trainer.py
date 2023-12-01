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
"""Trains transformer model for in-context learning."""

import functools
import os
from typing import Any, Callable, Mapping, Optional, Tuple

from absl import flags
from absl import logging
from flax import jax_utils
import flax.linen as nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from causallm_icl import plotting
from causallm_icl import predictor_flax
from causallm_icl import sampler_lib
from causallm_icl import transformer_lib_flax
from causallm_icl import utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow.io import gfile


flags.DEFINE_integer("n_epochs", default=5000, help="n_epochs")
flags.DEFINE_integer("n_iter_per_epoch", default=100, help="n_iter_per_epoch")
flags.DEFINE_float("learning_rate", default=1e-4, help="learnning_rate")
flags.DEFINE_float("weight_decay", default=0, help="weight_decay")
flags.DEFINE_string(
    "lr_scheduler_type", default="cosine", help="Use learning rate scheduler")
flags.DEFINE_float("adam_b1", default=0.9, help="Adam b1")
flags.DEFINE_float("adam_b2", default=0.999, help="Adam b2")
flags.DEFINE_float("adam_eps", default=1e-7, help="Adam eps")
flags.DEFINE_string(
    "x_distribution_str",
    default="normal*1.0+0.0",
    help="Training distribution for xs")
flags.DEFINE_string(
    "w_distribution_str",
    default="normal*1.0+0.0",
    help="Training distribution for ws")
flags.DEFINE_string("model_type", default="CausalLM", help="CausalLM/SimpleNN")
flags.DEFINE_float("noise_std", default=0.0, help="Noise std")
flags.DEFINE_float("test_scale", default=1.0, help="scale the test input")
flags.DEFINE_integer("eval_every_n_epochs", default=1000, help="eval_freq")
flags.DEFINE_integer("batch_train_data", default=1, help="in batches")
flags.DEFINE_string("output_map", default="linear", help="output mapping func")
flags.DEFINE_float("trh_lam", default=0.0, help="coeff for the trh term")
flags.DEFINE_bool("use_enc_mask", default=False,
                  help="apply the enc masks on in-context examples")
flags.DEFINE_bool("hidden_loss", default=False,
                  help="add the loss of hidden positions and layers")
flags.DEFINE_bool("even_loss", default=False,
                  help="add the loss of even positions")
flags.DEFINE_bool("permute_examples", default=False,
                  help="permute the in-context examples")
flags.DEFINE_integer("num_dec_examples", default=20,
                     help="number of decoder examples in the sequence")
flags.DEFINE_integer("num_classes", default=1,
                     help="number of classes, default 1 for linear regression")
flags.DEFINE_float("dropout_rate", default=0.0, help="dropout_rate")

FLAGS = flags.FLAGS


def train_step(state, seq, label, model, learning_rate_fn, dropout_rng=None):
  """Perform a single training step."""

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  num_dec_examples = FLAGS.num_dec_examples
  if FLAGS.permute_examples:
    l = seq.shape[-2]
    rand_order = jax.random.shuffle(
        dropout_rng, jnp.arange(l - num_dec_examples))
    rand_order = jnp.concatenate(
        [rand_order, jnp.arange(l-num_dec_examples, l)], axis=-1)
    seq = seq[..., rand_order, :]
    label = label[..., rand_order, :]

  def loss_fn(params):
    """loss function used for training."""
    start_step = label.shape[-2] - num_dec_examples
    errors, aux = model.apply({"params": params},
                              inputs=seq,
                              labels=label,
                              train=True,
                              rngs={"dropout": dropout_rng},
                              hidden_loss=FLAGS.hidden_loss,
                              use_enc_mask=FLAGS.use_enc_mask,
                              start_step=start_step)

    if FLAGS.even_loss:
      seq_loss = errors.mean()
    else:
      seq_loss = errors[:, start_step:].mean()

    return seq_loss, aux

  lr = learning_rate_fn(state.step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, extras), grads = grad_fn(state.params)  # extras is the aux in model.apply
  # See the returns of the call function of CausalLM for details.

  grads = jax.lax.pmean(grads, "batch")
  new_state = state.apply_gradients(grads=grads)
  loss = jax.lax.pmean(extras[0], "batch")
  y_errors = jax.lax.psum(extras[0], "batch").sum(axis=0)
  metrics = {"loss": loss, "lr": lr, "y_errors": y_errors}
  return new_state, metrics


def save_checkpoint(state, exp_folder: str) -> None:
  """Save model checkpoints."""

  def get_array(x):
    return np.array(x)

  state = jax.tree_util.tree_map(get_array, jax_utils.unreplicate(state))
  ckpt_dir = os.path.join(exp_folder, "ckpt/")
  gfile.makedirs(ckpt_dir)

  checkpoints.save_checkpoint(
      ckpt_dir=ckpt_dir, target=state, step=state.step, overwrite=True)


def get_model(rng: jax.Array, args: utils.ConfigDict):
  """Initialize model and optimizer states."""
  rng, init_rng = random.split(rng)

  config = transformer_lib_flax.TransformerConfig(
      num_heads=args.n_heads,
      num_layers=args.n_layers,
      num_classes=args.num_classes,
      hidden_size=args.hidden_size,
      loss_on_x_steps=args.loss_on_x_steps,
      norm_first=args.norm_first,
      disable_layer_norms=args.disable_layer_norms,
      final_layer_norm=args.final_layer_norm,
      dropout_rate=args.dropout_rate,
      attention_dropout_rate=args.dropout_rate,
      kernel_init=transformer_lib_flax.nn_init_parser(args.kernel_init),
      bias_init=transformer_lib_flax.nn_init_parser(args.bias_init),
      linear_w_init=transformer_lib_flax.nn_init_parser(args.linear_w_init),
      linear_bias_init=transformer_lib_flax.nn_init_parser(
          args.linear_bias_init),
      posemb_init=transformer_lib_flax.nn_init_parser(args.posemb_init),
      max_len=args.num_exemplars,
      activation_fn=transformer_lib_flax.nn_activation_parser(
          args.activation_fn),
      disable_softmax=args.disable_softmax,
      shared_block=args.shared_block,
  )

  if args.model_type == "CausalLM":
    model = predictor_flax.CausalLM(config)
  elif args.model_type == "SimpleNN":
    model = predictor_flax.SimpleNN(config)

  @jax.jit
  def initialize_variables(init_rng):
    init_batch = jnp.ones((1, config.max_len, args.x_dim + args.num_classes),
                          jnp.float32)
    init_variables = model.init(init_rng, inputs=init_batch, train=False)
    return init_variables

  init_variables = initialize_variables(init_rng)

  if args.lr_scheduler_type == "cosine":
    scheduler = transformer_lib_flax.create_learning_rate_scheduler(
        base_learning_rate=args.learning_rate,
        num_warmup_steps=(args.n_epochs // 5) * args.n_iter_per_epoch,
        num_training_steps=args.n_epochs * args.n_iter_per_epoch,
    )
  else:  # args.lr_scheduler_type == "warmup":
    scheduler = transformer_lib_flax.create_learning_rate_scheduler_v2(
        factors="constant * linear_warmup",
        base_learning_rate=args.learning_rate,
        warmup_steps=(args.n_epochs // 5) * args.n_iter_per_epoch,
    )

  opt = optax.adamw(
      scheduler,
      b1=args.adam_b1,
      b2=args.adam_b2,
      eps=args.adam_eps,
      weight_decay=args.weight_decay,
  )

  state = train_state.TrainState.create(
      apply_fn=model.apply, params=init_variables["params"], tx=opt)

  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, model=model, learning_rate_fn=scheduler),
      axis_name="batch",
  )

  return model, state, p_train_step


def eval_model(
    model,
    params,
    test_data,
    path: str,
):
  """Eval models on different distributions."""
  gfile.makedirs(path)

  seqs, coefficients, xs, ys = test_data
  seqs = jnp.array(seqs)

  if FLAGS.num_classes <= 1:
    plotting.plot_empirical_distribution(
        model,
        params,
        seqs,
        coefficients,
        xs=xs,
        ys=ys,
        ys_true=ys,
        prefix=path,
        start_step=FLAGS.num_exemplars - FLAGS.num_dec_examples,
        output_map=FLAGS.output_map,
        use_enc_mask=FLAGS.use_enc_mask,
    )
  else:
    plotting.plot_empirical_distribution_mc(
        model,
        params,
        seqs,
        ys_true=ys,
        prefix=path,
        start_step=FLAGS.num_exemplars - FLAGS.num_dec_examples,
        use_enc_mask=FLAGS.use_enc_mask,
    )


def train(
    rng: jax.Array,
    model: nn.Module,
    state: train_state.TrainState,
    p_train_step: Callable[
        ..., Tuple[train_state.TrainState, Mapping[str, Any]]
    ],
    exp_folder: Optional[str] = None,
    x_dim: int = 3,
    x_distribution_str: str = "normal*1+0",
    w_distribution_str: str = "normal*1+0",
    num_exemplars: int = 9,
    n_epochs: int = 100,
    n_iter_per_epoch: int = 100,
    batch_size: int = 64,
    noise_std: float = 0.0,
    final_xdim_one: bool = False,
):
  """Trains models."""
  _, new_rng = jax.random.split(rng)

  dropout_rngs = random.split(new_rng, jax.local_device_count())

  if FLAGS.num_classes <= 1:
    sampler = sampler_lib.LRSampler(
        num_exemplars,
        x_dim,
        cxw_gmm_fn=sampler_lib.str_to_gmm_mix_fn(x_distribution_str,
                                                 w_distribution_str),
        final_xdim_one=final_xdim_one,
        noise_std=noise_std,
        output_map=FLAGS.output_map,
    )
  else:
    sampler = sampler_lib.MCSampler(
        num_exemplars,
        x_dim,
        FLAGS.num_classes,
        cxw_gmm_fn=sampler_lib.str_to_gmm_mix_fn(x_distribution_str,
                                                 w_distribution_str),
        final_xdim_one=final_xdim_one,
        noise_std=noise_std,
    )

  train_data0 = sampler.sample(
      n=batch_size)
  if FLAGS.batch_train_data > 1:
    train_data1 = sampler.sample(n=batch_size * (FLAGS.batch_train_data - 1))
    train_seq = np.concatenate([train_data0[0], train_data1[0]], axis=0)
    train_label = np.concatenate([train_data0[-1], train_data1[-1]], axis=0)
  else:
    train_seq = train_data0[0]
    train_label = train_data0[-1]

  test_data = sampler.sample(
      n=batch_size, scale=FLAGS.test_scale)

  num_train = train_seq.shape[0]
  metrics_full = []
  for epoch in range(n_epochs):
    metrics_all = []

    for _ in range(n_iter_per_epoch):
      train_batch_ids = np.random.choice(num_train, batch_size, replace=False)
      unshard_seqs = jnp.array(train_seq[train_batch_ids])
      seqs = common_utils.shard(unshard_seqs)

      unshard_labels = jnp.array(train_label[train_batch_ids])
      labels = common_utils.shard(unshard_labels)

      state, metrics = p_train_step(state, seqs, labels,
                                    dropout_rng=dropout_rngs)
      metrics = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))
      metrics_all.append(metrics)
      metrics_full.append(metrics)

    if exp_folder is not None:
      if epoch == n_epochs - 1 or (epoch + 1) % FLAGS.eval_every_n_epochs == 0:
        eval_model(
            model,
            jax_utils.unreplicate(state).params,
            test_data,
            path=f"{exp_folder}/plots/{epoch+1}/test/",
        )
        eval_model(
            model,
            jax_utils.unreplicate(state).params,
            train_data0,
            path=f"{exp_folder}/plots/{epoch+1}/train/",
        )

    if epoch % 100 == 0:
      metrics_all = common_utils.stack_forest(metrics_all)
      y_errors = jnp.mean(metrics_all["y_errors"], axis=0) / batch_size
      logging.info("Epoch %d is finished.", epoch)
      logging.info(y_errors)

  metrics_full = common_utils.stack_forest(metrics_full)
  metrics_full["y_errors"] = jnp.mean(
      metrics_full["y_errors"], axis=0) / batch_size

  return state, metrics_full
