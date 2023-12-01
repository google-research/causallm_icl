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
"""Tests for model_trainer."""


from absl.testing import absltest

from causallm_icl import model_trainer
from causallm_icl import utils
from jax import random


class ModelTrainerTest(absltest.TestCase):

  def test_model_trainer_pipeline(self):

    args = utils.flags_to_args()

    args.seed = 0
    args.batch_size = 1
    args.x_dim = 2
    args.n_layers = 3
    args.num_exemplars = 40
    args.hidden_size = 8
    args.n_heads = 2
    args.n_iter_per_epoch = 1
    args.n_epochs = 1
    args.num_classes = 1
    args.dropout_rate = 0.0

    utils.set_seed(args.seed)
    rng = random.PRNGKey(args.seed)
    _, new_rng = random.split(rng)

    model, state, p_train_step = model_trainer.get_model(new_rng, args)

    _, metrics = model_trainer.train(
        new_rng,
        model,
        state,
        p_train_step,
        exp_folder=None,
        num_exemplars=args.num_exemplars,
        n_epochs=args.n_epochs,
        x_dim=args.x_dim,
        n_iter_per_epoch=args.n_iter_per_epoch,
        batch_size=args.batch_size)
    print(metrics)
    self.assertAlmostEqual(metrics['y_errors'][-2], 6.17171764e-01, places=5)
    self.assertAlmostEqual(metrics['y_errors'][-1], 9.07987356e-01, places=5)


if __name__ == '__main__':
  absltest.main()
