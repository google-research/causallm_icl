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
"""Plot the theoretical figures in Exp-1."""
from absl import app
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.io import gfile


def main(_):
  plt.rcParams.update({'font.size': 12})
  noise = 0
  prefix = '/tmp/'
  draw_full_converge(noise, prefix)
  draw_autoregressive_converge(noise, prefix)
  draw_autoregressive_stationary(noise, prefix)


def draw_full_stationary(noise=0, prefix=''):
  """Draw autoregressive stationary test error."""

  if noise > 0.:
    prefix += f'ns{noise}_'
  fpath = prefix + 'prefixlm_stationary_test_error.pdf'
  ax = plt.axes()
  ax.set_title('Test Error of the causalLM-ICL Stationary Point')
  ax.set_ylabel('Test MSE')
  ax.set_xlabel('Number of training exemplars')
  ax.set_yscale('log')

  max_train_length = 600
  test_length = 200
  length = max_train_length + test_length
  dim = 16
  batch = 1

  for offset in [-1, 0, 1, 2]:
    x = np.random.uniform(size=[batch, length, dim]) * 2. + offset
    w = np.random.normal(size=[batch, dim])
    y = np.einsum('bli,bi->bl', x, w)
    y[:, :max_train_length] += noise * np.random.normal(
        size=[batch, max_train_length])

    train_lengths = list(range(20, max_train_length+1, 20))
    test_errors = []
    for _ in train_lengths:
      w = np.linalg.pinv(x[0]) @ y[0]
      ym = np.zeros([length])
      test_error = 0.
      for m in range(max_train_length + 1, length+1):
        ym[m-1] += np.dot(w, x[0, m-1, :])
        test_error += (y[0, m-1] - ym[m-1]) ** 2
      test_error /= test_length
      test_errors.append(test_error)

    ax.plot(train_lengths, test_errors)

  ax.legend([r'$\mu_x=0$', r'$\mu_x=1$', r'$\mu_x=2$', r'$\mu_x=3$'])
  with gfile.GFile(fpath, 'wb') as handle:
    plt.savefig(handle, format='pdf')
  plt.close()


def draw_autoregressive_stationary(noise=0, prefix=''):
  """Draw autoregressive stationary test error."""

  if noise > 0.:
    prefix += f'ns{noise}_'
  fpath = prefix + 'causallm_stationary_test_error.pdf'
  ax = plt.axes()
  ax.set_title('Stationary points of causalLM-ICL')
  ax.set_ylabel('Query MSE error')
  ax.set_xlabel('Number of training exemplars')
  ax.set_yscale('log')

  max_train_length = 300
  test_length = 200
  length = max_train_length + test_length
  dim = 16
  batch = 64

  for offset in [-1, 0, 1, 2]:
    x = np.random.uniform(size=[batch, length, dim]) * 2. + offset
    w = np.random.normal(size=[batch, dim])
    y = np.einsum('bli,bi->bl', x, w)
    y[:, :max_train_length] += noise * np.random.normal(
        size=[batch, max_train_length])

    xtx = np.einsum('bli,bmi->blm', x, x)

    a = np.zeros([batch, max_train_length+1])
    for j in range(1, max_train_length+1):
      a[:, j] = y[:, j-1]
      for i in range(1, j):
        a[:, j] -= a[:, i] * xtx[:, i-1, j-1]
      a[:, j] /= xtx[:, j-1, j-1]

    train_lengths = list(range(20, max_train_length+1, 20))
    test_errors = []
    for train_length in train_lengths:
      ym = np.zeros([batch, length])
      test_error = 0.
      for m in range(max_train_length + 1, length+1):
        for i in range(1, train_length+1):
          ym[:, m-1] += a[:, i] * xtx[:, i-1, m-1]
        test_error += np.mean((y[:, m-1] - ym[:, m-1]) ** 2)
      test_error /= test_length
      test_errors.append(test_error)

    ax.plot(train_lengths, test_errors, linewidth=4)

  ax.legend([r'$\mu_x=0$', r'$\mu_x=1$', r'$\mu_x=2$', r'$\mu_x=3$'])
  with gfile.GFile(fpath, 'wb') as handle:
    plt.savefig(handle, format='pdf')
  plt.close()


def draw_autoregressive_converge(noise=0, prefix=''):
  """Draw autoregressive converge speed."""
  length = 60
  dim = 16
  batch = 64
  x = np.random.uniform(size=[batch, length, dim]) * 2. - 1
  w = np.random.normal(size=[batch, dim])
  y = np.einsum('bli,bi->bl', x, w)

  train_length = 40
  test_length = length - train_length
  y[:, :train_length] += noise * np.random.normal(size=[batch, train_length])
  nit = 50
  eta = 0.04
  a = np.zeros([batch, train_length+1, nit+1])
  train_error = np.zeros([nit+1])
  test_error = np.zeros([nit+1])

  plot_step = 2

  xtx = np.einsum('bli,bmi->blm', x, x)

  for it in range(1, nit+1):
    for j in range(1, train_length+1):
      a[:, j, it] = a[:, j, it-1] + eta * y[:, j-1]
      for k in range(1, j+1):
        a[:, j, it] -= eta * a[:, k, it-1] * xtx[:, k-1, j-1]

    ym = np.zeros([batch, length])
    for m in range(1, train_length+1):
      for i in range(1, m+1):
        ym[:, m-1] += a[:, i, it] * xtx[:, i-1, m-1]
      train_error[it] += np.mean((y[:, m-1] - ym[:, m-1]) ** 2)
    train_error[it] /= train_length

    for m in range(train_length + 1, length+1):
      for i in range(1, train_length+1):
        ym[:, m-1] += a[:, i, it] * xtx[:, i-1, m-1]
      test_error[it] += np.mean((y[:, m-1] - ym[:, m-1]) ** 2)
    test_error[it] /= test_length

  if noise > 0.:
    prefix += f'ns{noise}_'
  fpath = prefix + 'causallm_converge_speed.pdf'
  plt.plot(range(1, nit+1, plot_step), train_error[1::plot_step],
           'r*', linewidth=2, markersize=12)
  plt.plot(range(1, nit+1, plot_step), test_error[1::plot_step],
           'r+', linewidth=2, markersize=12)
  plt.title('Convergence of multi-layer causalLM-ICL')
  plt.legend(['In-context examples', 'Query examples'])
  plt.yscale('log')
  plt.xlabel('Number of layers')
  plt.ylabel('MSE Error')
  with gfile.GFile(fpath, 'wb') as handle:
    plt.savefig(handle, format='pdf')
  plt.close()


def draw_full_converge(noise=0, prefix=''):
  """Draw full converge speed."""
  length = 60
  dim = 16
  batch = 64
  x = np.random.uniform(size=[batch, length, dim]) * 2. - 1
  w = np.random.normal(size=[batch, dim])
  y = np.einsum('bli,bi->bl', x, w)
  y += noise * np.random.normal(size=[batch, length])

  train_length = 40
  test_length = length - train_length
  y[:, :train_length] += noise * np.random.normal(size=[batch, train_length])
  nit = 50
  eta = 0.04
  a = np.zeros([batch, train_length+1, nit+1])
  train_error = np.zeros([nit+1])
  test_error = np.zeros([nit+1])

  plot_step = 2

  xtx = np.einsum('bli,bmi->blm', x, x)

  for it in range(1, nit+1):
    for j in range(1, train_length+1):
      a[:, j, it] = a[:, j, it-1] + eta * y[:, j-1]
      for k in range(1, train_length+1):
        a[:, j, it] -= eta * a[:, k, it-1] * xtx[:, k-1, j-1]

    ym = np.zeros([batch, length])
    for m in range(1, train_length+1):
      for i in range(1, train_length+1):
        ym[:, m-1] += a[:, i, it] * xtx[:, i-1, m-1]
      train_error[it] += np.mean((y[:, m-1] - ym[:, m-1]) ** 2)
    train_error[it] /= train_length

    for m in range(train_length + 1, length+1):
      for i in range(1, train_length+1):
        ym[:, m-1] += a[:, i, it] * xtx[:, i-1, m-1]
      test_error[it] += np.mean((y[:, m-1] - ym[:, m-1]) ** 2)
    test_error[it] /= test_length

  if noise > 0.:
    prefix += f'ns{noise}_'
  fpath = prefix + 'prefixlm_converge_speed.pdf'
  plt.plot(range(1, nit+1, plot_step), train_error[1::plot_step],
           'b*', linewidth=2, markersize=12)
  plt.plot(range(1, nit+1, plot_step), test_error[1::plot_step],
           'bo', linewidth=2, markersize=12)
  plt.title('Convergence of multi-layer prefixLM-ICL')
  plt.legend(['In-context examples', 'Query examples'])
  plt.yscale('log')
  plt.xlabel('Number of layers')
  plt.ylabel('MSE Error')
  with gfile.GFile(fpath, 'wb') as handle:
    plt.savefig(handle, format='pdf')
  plt.close()


if __name__ == '__main__':
  app.run(main)
