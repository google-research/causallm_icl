# PACTran Metrics

This is the code repository for the paper: CausalLM is not optimal for in-context learning (`https://arxiv.org/abs/2308.06912`).

## Introduction

Recent empirical evidence indicates that transformer based in-context learning performs better when using a prefix language model (prefixLM), in which in-context samples can all attend to each other, compared to causal language models (causalLM), which use auto-regressive attention that prohibits in-context samples to attend to future samples. While this result is intuitive, it is not understood from a theoretical perspective. In this paper we take a theoretical approach and analyze the convergence behavior of prefixLM and causalLM under a certain parameter construction. Our analysis shows that both LM types converge to their stationary points at a linear rate, but that while prefixLM converges to the optimal solution of linear regression, causalLM convergence dynamics follows that of an online gradient descent algorithm, which is not guaranteed to be optimal even as the number of samples grows infinitely. We supplement our theoretical claims with empirical experiments over synthetic and real tasks and using various types of transformers. Our experiments verify that causalLM consistently underperforms prefixLM in all settings.

## How to use

This repository currently contain the code for Sec 5.1 the LSA-transformers on linear regression problems, and Sec 5.2 ordinary softmax transformers on synthetic problems.

- Prerequisites:
  - Tensorflow
  - JAX
  - Flax
  - Numpy
  - Scipy

- For the LSA-transformers, run
python theory_plots.py

- For the ordinary softmax transformers, run

exp_folder=       # experiment output location
use_enc_mask=     # True: prefixLM, False: CausalLM
num_classes=      # 1: regression, >1: classification
output_map=       # 'linear' for linear regression or classification, 'sigmoid' for non-linear regression
shared_block=     # True: Shared-block transformer, False: not sharing layers
num_exemplars=    # number of training examples + testing examples (default testing 20) in each sequence

# The complete set of flags are listed in main.py, model_trainer.py, transformer_lib_flax.py

python main.py -- \
--exp_folder ${exp_folder} \
--use_enc_mask ${use_enc_mask} \
--num_clases ${num_classes} \
--output_map ${output_map} \
--shared_block ${shared_block} \
--num_exemplars ${num_exemplars}
