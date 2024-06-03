# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any

import numpy as np
import logging
import functools
from models import ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS

def train(config, workdir):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    # Initialize model.
    score_model, init_model_state, initial_params = mutils.init_model(config)

    optimizer = losses.get_optimizer(config)(score_model.parameters())

    state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                         model_state=init_model_state,
                         ema_rate=config.model.ema_rate,
                         params_ema=initial_params,
                         rng=None)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_meta_dir, exist_ok=True)
    state = utils.restore_checkpoint(checkpoint_meta_dir, state)
    initial_step = int(state.step)

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=config.training.n_jitted_steps,
                                                uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size // torch.cuda.device_count(), config.data.image_size,
                          config.data.image_size, config.data.num_channels)
        sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

    pstate = state
    num_train_steps = config.training.n_iters

    logging.info("Starting training loop at step %d." % (initial_step,))

    n_jitted_steps = config.training.n_jitted_steps
    assert config.training.log_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
           config.training.eval_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

    for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
        batch = {k: scaler(torch.tensor(v.numpy())) for k, v in next(train_iter).items()}
        pstate, ploss = train_step_fn(pstate, batch)
        loss = ploss.mean()

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss.item(), step)

        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            saved_state = pstate
            utils.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1)

        if step % config.training.eval_freq == 0:
            eval_batch = {k: scaler(torch.tensor(v.numpy())) for k, v in next(eval_iter).items()}
            pstate, peval_loss = eval_step_fn(pstate, eval_batch)
            eval_loss = peval_loss.mean()
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            saved_state = pstate
            utils.save_checkpoint(checkpoint_dir, saved_state,
                                  step=step // config.training.snapshot_freq,
                                  keep=np.inf)

            if config.training.snapshot_sampling:
                sample, n = sampling_fn(pstate)
                this_sample_dir = os.path.join(sample_dir, "iter_{}_host_{}".format(step, 0))
                os.makedirs(this_sample_dir, exist_ok=True)
                image_grid = sample.reshape((-1, *sample.shape[2:]))
                nrow = int(np.sqrt(image_grid.shape[0]))
                sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
                with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, sample)

                with open(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    utils.save_image(image_grid, fout, nrow=nrow, padding=2)

def evaluate(config, workdir, device, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        device: The device to run the evaluation on.
        eval_folder: The subfolder for storing evaluation results. Default to "eval".
    """
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    torch.manual_seed(config.seed + 1)

    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=1,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    score_model, init_model_state, initial_params = mutils.init_model(config)
    score_model.to(device)
    optimizer = losses.get_optimizer(config)(score_model.parameters())
    state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                         model_state=init_model_state,
                         ema_rate=config.model.ema_rate,
                         params_ema=initial_params,
                         rng=None)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, score_model,
                                       train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean,
                                       continuous=continuous, likelihood_weighting=likelihood_weighting)

    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        additional_dim=None,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
        ds_bpd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler)

    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size // torch.cuda.device_count(),
                          config.data.image_size, config.data.image_size,
                          config.data.num_channels)
        sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

    class EvalMeta:
        def __init__(self, ckpt_id, sampling_round_id, bpd_round_id, rng):
            self.ckpt_id = ckpt_id
            self.sampling_round_id = sampling_round_id
            self.bpd_round_id = bpd_round_id
            self.rng = rng

    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
    num_bpd_rounds = len(ds_bpd) * bpd_num_repeats

    eval_meta = EvalMeta(config.eval.begin_ckpt, -1, -1, None)
    eval_meta = utils.restore_checkpoint(
        eval_dir, eval_meta, step=None, prefix=f"meta_0_")

    if eval_meta.bpd_round_id < num_bpd_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = eval_meta.bpd_round_id + 1
        begin_sampling_round = 0

    elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = num_bpd_rounds
        begin_sampling_round = eval_meta.sampling_round_id + 1

    else:
        begin_ckpt = eval_meta.ckpt_id + 1
        begin_bpd_round = 0
        begin_sampling_round = 0

    rng = eval_meta.rng
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        try:
            state = utils.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        except:
            time.sleep(60)
            try:
                state = utils.restore_checkpoint(checkpoint_dir, state, step=ckpt)
            except:
                time.sleep(120)
                state = utils.restore_checkpoint(checkpoint_dir, state, step=ckpt)

        pstate = state
        if config.eval.enable_loss:
            all_losses = []
            eval_iter = iter(eval_ds)
            for i, batch in enumerate(eval_iter):
                eval_batch = {k: scaler(torch.tensor(v.numpy())).to(device) for k, v in batch.items()}
                pstate, p_eval_loss = eval_step(pstate, eval_batch)
                eval_loss = p_eval_loss.mean()
                all_losses.append(eval_loss)
                if (i + 1) % 1000 == 0:
                    logging.info("Finished %dth step loss evaluation" % (i + 1))

            all_losses = np.asarray(all_losses)
            with open(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
                fout.write(io_buffer.getvalue())

        if config.eval.enable_bpd:
            bpds = []
            begin_repeat_id = begin_bpd_round // len(ds_bpd)
            begin_batch_id = begin_bpd_round % len(ds_bpd)
            for repeat in range(begin_repeat_id, bpd_num_repeats):
                bpd_iter = iter(ds_bpd)
                for _ in range(begin_batch_id):
                    next(bpd_iter)
                for batch_id in range(begin_batch_id, len(ds_bpd)):
                    batch = next(bpd_iter)
                    eval_batch = {k: scaler(torch.tensor(v.numpy())).to(device) for k, v in batch.items()}
                    bpd = likelihood_fn(pstate, eval_batch['image'])[0]
                    bpd = bpd.reshape(-1)
                    bpds.append(bpd)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    with open(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"), "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, bpd)
                        fout.write(io_buffer.getvalue())

                    eval_meta.ckpt_id = ckpt
                    eval_meta.bpd_round_id = bpd_round_id
                    eval_meta.rng = rng
                    utils.save_checkpoint(
                        eval_dir,
                        eval_meta,
                        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + bpd_round_id,
                        keep=1,
                        prefix=f"meta_0_")
        else:
            eval_meta.ckpt_id = ckpt
            eval_meta.bpd_round_id = num_bpd_rounds - 1
            utils.save_checkpoint(
                eval_dir,
                eval_meta,
                step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_bpd_rounds - 1,
                keep=1,
                prefix=f"meta_0_")

        if config.eval.enable_sampling:
            state = state
            for r in range(begin_sampling_round, num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
                this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_0")
                os.makedirs(this_sample_dir, exist_ok=True)

                sample, n = sampling_fn(pstate)
                samples = np.clip(sample * 255., 0, 255).astype(np.uint8)
                samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                with open(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                gc.collect()
                latents = evaluation.run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
                gc.collect()
                with open(os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
                    fout.write(io_buffer.getvalue())

                eval_meta.ckpt_id = ckpt
                eval_meta.sampling_round_id = r
                eval_meta.rng = rng
                if r < num_sampling_rounds - 1:
                    utils.save_checkpoint(
                        eval_dir,
                        eval_meta,
                        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
                        keep=1,
                        prefix=f"meta_0_")

            all_logits = []
            all_pools = []
            for host in range(1):  # Assuming single host
                this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")

                stats = os.listdir(this_sample_dir)
                wait_message = False
                while len(stats) < num_sampling_rounds:
                    if not wait_message:
                        logging.warning("Waiting for statistics on host %d" % (host,))
                        wait_message = True
                    stats = os.listdir(this_sample_dir)
                    time.sleep(30)

                for stat_file in stats:
                    with open(os.path.join(this_sample_dir, stat_file), "rb") as fin:
                        stat = np.load(fin)
                        if not inceptionv3:
                            all_logits.append(stat["logits"])
                        all_pools.append(stat["pool_3"])

            if not inceptionv3:
                all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
            all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

            data_stats = evaluation.load_dataset_stats(config)
            data_pools = data_stats["pool_3"]

            if not inceptionv3:
                inception_score = evaluation.classifier_score_from_logits(all_logits)
            else:
                inception_score = -1

            fid = evaluation.frechet_classifier_distance_from_activations(data_pools, all_pools)
            kid = evaluation.kernel_classifier_distance_from_activations(data_pools, all_pools)

            logging.info(
                "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
                    ckpt, inception_score, fid, kid))

            with open(os.path.join(eval_dir, f"report_{ckpt}.npz"), "wb") as f:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
                f.write(io_buffer.getvalue())

            utils.save_checkpoint(
                eval_dir,
                eval_meta,
                step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
                keep=1,
                prefix=f"meta_0_")

        else:
            eval_meta.ckpt_id = ckpt
            eval_meta.sampling_round_id = num_sampling_rounds - 1
            eval_meta.rng = rng
            utils.save_checkpoint(
                eval_dir,
                eval_meta,
                step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_sampling_rounds - 1 + num_bpd_rounds,
                keep=1,
                prefix=f"meta_0_")

        begin_bpd_round = 0
        begin_sampling_round = 0

    meta_files = os.listdir(eval_dir)
    for file in meta_files:
        if file.startswith("meta_0_"):
            os.remove(os.path.join(eval_dir, file))
