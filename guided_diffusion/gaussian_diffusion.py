# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum

import numpy as np
import torch
import torch as th
from utils import Blur_Shapen
from collections import defaultdict
import os

from PIL import Image

from guided_diffusion.scheduler import get_schedule_jump
from utils.File_Utils import make_dirs


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, use_scale):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.

        if use_scale:
            scale = 1000 / num_diffusion_timesteps
        else:
            scale = 1

        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
            conf=None
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.conf = conf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_prev_prev = np.append(
            1.0, self.alphas_cumprod_prev[:-1])

        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.betas_multi_rec_alphas_cumprod = self.betas / self.alphas_cumprod
        self.betas_multi_rec_alphas_cumprod_cumsum = np.cumsum(self.betas_multi_rec_alphas_cumprod)
        self.betas_multi_rec_alphas_cumprod_cumsum_pre = np.append(0.0, self.betas_multi_rec_alphas_cumprod_cumsum)

        # print(self.sqrt_alphas_cumprod[1]*(self.sqrt_betas_multi_rec_alphas_cumprod_cumsum[1]-self.sqrt_betas_multi_rec_alphas_cumprod_cumsum_pre[0]), np.sqrt(1-self.alphas_cumprod[1]))

        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) /
                (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) /
                (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        # print(self.alphas_cumprod)

    def undo(self, image_before_step, img_after_model, est_x_0, model_kwargs, t, conf, eps, last_log_variance, jump, debug=False):
        return self._undo(img_after_model, est_x_0, model_kwargs, t, last_log_variance=last_log_variance, eps=eps, conf=conf, jump=jump)

    def _undo(self, img_after_model, est_x_0, model_kwargs, t, conf, eps, last_log_variance, jump):
        # blur x0 predicted by xt
        # blur = Blur.GaussianBlur(channels=3, kernel_size=5, sigma=(t.item() / 250)).cuda()
        # alphas_comprod = _extract_into_tensor(self.alphas_cumprod, t, est_x_0.shape)
        # img_in_est = blur(est_x_0)
        # img_in_est = th.sqrt(alphas_comprod) * img_in_est + \
        #     th.sqrt(1-alphas_comprod) * th.randn_like(img_in_est)

        # repaint(xt+10 from xt)
        img_in_est = img_after_model
        for i in range(jump):
            beta = _extract_into_tensor(self.betas, t - jump + i + 1, img_after_model.shape)
            img_in_est = th.sqrt(1 - beta) * img_in_est + \
                         th.sqrt(beta) * th.randn_like(img_in_est)

        # # repaint with eps
        # img_in_est = img_after_model
        # for i in range(jump):
        #     beta = _extract_into_tensor(self.betas, t - jump + i + 1, img_after_model.shape)
        #     img_in_est = th.sqrt(1 - beta) * img_in_est + \
        #                  th.sqrt(beta) * eps

        # repaint(xt+10 from x0)
        # alphas_comprod_plus_jump_length = _extract_into_tensor(self.alphas_cumprod, t, est_x_0.shape)
        # img_in_est = th.sqrt(alphas_comprod_plus_jump_length) * est_x_0 + \
        #     th.sqrt(1 - alphas_comprod_plus_jump_length) * eps

        # debug
        # print(t)
        # img_in_est = est_x_0
        # for i in range(t.item()):
        #     sqrt_betas = to_tensor(self.sqrt_alphas_cumprod, partial.device)[t] * (
        #                     to_tensor(self.sqrt_betas_multi_rec_alphas_cumprod_cumsum, partial.device)[t] -
        #                     to_tensor(self.sqrt_betas_multi_rec_alphas_cumprod_cumsum, partial.device)[t-jump])
        #     tmp_betas = (th.exp(0.5 * last_log_variance))
        #     img_in_est = img_in_est * th.sqrt(_extract_into_tensor(1.0 - self.betas, th.full_like(t, i), est_x_0.shape))\
        #                  + tmp_betas * th.randn_like(est_x_0)
        #     print('sss')
                         # + th.sqrt(_extract_into_tensor(self.betas, th.full_like(t, i), est_x_0.shape)) \

        # anti_partial repaint
        # partial = model_kwargs['weight_mask']
        # partial = th.floor(partial * (t - jump))
        # partial = th.where(partial < 0, 0, partial)
        # partial = partial.to(th.long)
        # img_mid = est_x_0 * th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[partial]) \
        #           + eps * th.sqrt(to_tensor(1 - self.alphas_cumprod, partial.device)[partial])
        # sqrt_alphas = th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[t].expand(partial.shape) /
        #                       to_tensor(self.alphas_cumprod, partial.device)[partial])
        # sqrt_betas = to_tensor(self.sqrt_alphas_cumprod, partial.device)[t] * (
        #     th.sqrt(to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[t] -
        #             to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[partial]))
        # beta = _extract_into_tensor(1 - self.alphas_cumprod, t, est_x_0.shape)
        # img_in_est = sqrt_alphas * img_mid + sqrt_betas * th.randn_like(est_x_0)

        # partial repaint
        # partial = 1 - model_kwargs['weight_mask']
        # partial = th.floor(partial * (t - jump))
        # partial = th.where(partial < 0, 0, partial)
        # partial = partial.to(th.long)
        # img_mid = est_x_0 * th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[partial]) \
        #           + eps * th.sqrt(to_tensor(1 - self.alphas_cumprod, partial.device)[partial])
        # sqrt_alphas = th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[t].expand(partial.shape) /
        #                       to_tensor(self.alphas_cumprod, partial.device)[partial])
        # sqrt_betas = to_tensor(self.sqrt_alphas_cumprod, partial.device)[t] * (
        #     th.sqrt(to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[t] -
        #             to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[partial]))
        # beta = _extract_into_tensor(1 - self.alphas_cumprod, t, est_x_0.shape)
        # img_in_est = sqrt_alphas * img_mid + sqrt_betas * th.randn_like(est_x_0)

        # repaint(xt+10 from x0) introduce weight_mask
        # if(t>=0):
        # weight_mask = model_kwargs['weight_mask'] if t < 2000 else th.ones_like(est_x_0)
        # alphas_comprod_plus_jump_length = _extract_into_tensor(self.alphas_cumprod, t, est_x_0.shape)
        # img_in_est = th.sqrt(alphas_comprod_plus_jump_length) * est_x_0 + \
        #              th.sqrt(1 - alphas_comprod_plus_jump_length) * th.randn_like(est_x_0) \
        #     * weight_mask

        # blur compose image of gt * mask + x0 * (1-mask)
        # blur = Blur_Shapen.GaussianBlur(channels=3, kernel_size=5, sigma=(t.item() / 125) ** 2).cuda()
        # gt_keep_mask = model_kwargs.get('gt_keep_mask')
        # gt = model_kwargs['gt']
        # compose = gt * gt_keep_mask + est_x_0 * (1-gt_keep_mask)
        # alphas_comprod = _extract_into_tensor(self.alphas_cumprod, t, compose.shape)
        # compose = blur(compose)
        # img_in_est = th.sqrt(alphas_comprod) * compose + \
        #     th.sqrt(1-alphas_comprod) * th.randn_like(compose)

        # shapen compose image of gt * mask + x0 * (1-mask)
        # shapen = Blur_Shapen.Sharpen(0.1 * t.item() / 250).cuda()
        # gt_keep_mask = model_kwargs.get('gt_keep_mask')
        # gt = model_kwargs['gt']
        # compose = gt * gt_keep_mask + est_x_0 * (1 - gt_keep_mask)
        # alphas_comprod = _extract_into_tensor(self.alphas_cumprod, t, compose.shape)
        # # if t.item() < 100:
        # #     compose = torch.clamp(shapen(compose), 0, 1)
        # if t.item() > 100:
        #     compose = blur(compose)
        # img_in_est = th.sqrt(alphas_comprod) * compose + \
        #              th.sqrt(1 - alphas_comprod) * th.randn_like(compose)

        return img_in_est

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1,
                                     t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2,
                                       t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "eps": model_output,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(
                    self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] *
                gradient.float()
        )
        return new_mean

    def p_sample(
            self,
            model,
            x,
            t,
            fix_noise,
            image_id,
            counter,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            conf=None,
            meas_fn=None,
            pred_xstart=None,
            idx_wall=-1,

    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        noise = th.randn_like(x)
        if conf.inpa_inj_sched_prev:

            if pred_xstart is not None:
                gt_keep_mask = model_kwargs.get('gt_keep_mask')
                if gt_keep_mask is None:
                    gt_keep_mask = conf.get_inpa_mask(x)

                gt = model_kwargs['gt']

                alpha_cumprod = _extract_into_tensor(
                    self.alphas_cumprod, t, x.shape)

                if conf.inpa_inj_sched_prev_cumnoise:
                    weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
                else:
                    gt_weight = th.sqrt(alpha_cumprod)
                    gt_part = gt_weight * gt
                    ################固定噪声
                    noise_weight = th.sqrt((1 - alpha_cumprod))
                    # noise_part = noise_weight * fix_noise
                    ################
                    noise_part = noise_weight * th.randn_like(x)

                    weighed_gt = gt_part + noise_part

                # directory = conf.pget('data.eval.paper_face_mask.paths.reverse_processing') + '/' + str(
                #     image_id)
                # make_dirs(directory)
                # # subfold = 'withp' if conf.pget('withp') else 'withoutp'
                # img = weighed_gt
                # img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
                # img = img.permute(0, 2, 3, 1)
                # img = img.contiguous().squeeze()
                # img = img.cpu().numpy()
                # img = Image.fromarray(img, mode='RGB')
                # full_p1 = os.path.join(directory, 'gt_noise' + '_' + str(counter).zfill(6) + '.jpg')
                # img.save(full_p1)

                x = (
                        gt_keep_mask * (
                    weighed_gt
                )
                        +
                        (1 - gt_keep_mask) * (
                            x
                        )
                )

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * \
                 th.exp(0.5 * out["log_variance"]) * noise
        if t > 0:
            if not conf["new"]:
                # time_based
                partial = model_kwargs['weight_mask'] * (0.8 + 0.2 * (1 - t/int(conf.pget("schedule_jump_params.t_T"))))
                partial = partial * (t-2)
                partial = th.where(partial < 0, 0, partial)
                partial = partial.to(th.long)

                ######################################################################### assumption2
                # partial = model_kwargs['weight_mask']
                # partial = partial.fill_(0.85 * t.item()).to(th.long)
                # partial = th.where(partial < 0, 0, partial)
                # width = conf.pget('assumption2.width')
                # height = conf.pget('assumption2.height')
                # startx = conf.pget('assumption2.startx')
                # starty = conf.pget('assumption2.starty')
                # index = (0, slice(0, 3), slice(startx, startx + height), slice(starty, starty + width))
                # partial[index] = t - 1
                #########################################################################

                img_mid = out["pred_xstart"] * th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[partial]) \
                          + out["eps"] * th.sqrt(to_tensor(1 - self.alphas_cumprod, partial.device)[partial])
                sqrt_alphas = th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[t-1].expand(partial.shape) /
                                      to_tensor(self.alphas_cumprod, partial.device)[partial])
                sqrt_betas = to_tensor(self.sqrt_alphas_cumprod, partial.device)[t-1] * (
                    th.sqrt(to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[t-1] -
                            to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[partial]))
                sample = sqrt_alphas * img_mid + 1.2 * sqrt_betas * th.randn_like(img_mid)
        # ############################################################

        # ######################################### new sampling(t smaller, mid bigger)
            if conf["new"]:
                partial = model_kwargs['weight_mask'] * (0.5 + 0.5 * (1 - t/int(conf.pget("schedule_jump_params.t_T"))))
                partial = th.floor(partial * (t - 2))
                partial = th.where(partial < 0, 0, partial)
                partial = partial.to(th.long)
                img_mid = out["pred_xstart"] * th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[partial]) \
                          + out["eps"] * th.sqrt(to_tensor(1 - self.alphas_cumprod, partial.device)[partial])
                sqrt_alphas = th.sqrt(to_tensor(self.alphas_cumprod, partial.device)[t - 1].expand(partial.shape) /
                                      to_tensor(self.alphas_cumprod, partial.device)[partial])
                sqrt_betas = to_tensor(self.sqrt_alphas_cumprod, partial.device)[t - 1] * (
                    th.sqrt(to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[t - 1] -
                            to_tensor(self.betas_multi_rec_alphas_cumprod_cumsum, partial.device)[partial]))
                sample = sqrt_alphas * img_mid + sqrt_betas * th.randn_like(img_mid)
        # ############################################################
            # directory = conf.pget('data.eval.paper_face_mask.paths.reverse_processing') + '/' + str(
            #     image_id)
            # make_dirs(directory)
            # # subfold = 'withp' if conf.pget('withp') else 'withoutp'
            # img = img_mid
            # img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
            # img = img.permute(0, 2, 3, 1)
            # img = img.contiguous().squeeze()
            # img = img.cpu().numpy()
            # img = Image.fromarray(img, mode='RGB')
            # full_p1 = os.path.join(directory, 'mid' + '_' + str(counter).zfill(6) + '.jpg')
            # img.save(full_p1)

        # ######################################### DDIM
        # if t > 0:
        #     alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        #     alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        #     eta = 0.5
        #     sigma = (
        #             eta
        #             * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        #             * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        #     )
        #     # print(sigma)
        #     noise = th.randn_like(x)
        #     sample = (
        #         out["pred_xstart"] * th.sqrt(alpha_bar_prev)
        #         + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * out["eps"]
        #     )
        #     nonzero_mask = (
        #         (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        #     )  # no noise when t == 0
        #     sample = sample + nonzero_mask * sigma * noise
        # ############################################################
        result = {"sample": sample,
                  "pred_xstart": out["pred_xstart"],
                  'gt': model_kwargs.get('gt'),
                  'eps': out['eps'],
                  'log_variance': out["log_variance"]}
        return result

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            return_all=False,
            conf=None,
            image_id=1
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                conf=conf,
                image_id=image_id
        ):
            final = sample

        if return_all:
            return final
        else:
            return final["sample"]

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            conf=None,
            image_id=1
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        debug_steps = conf.pget('debug.num_timesteps')

        self.gt_noises = None  # reset for next image

        pred_xstart = None

        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        if conf.schedule_jump_params:
            times = get_schedule_jump(**conf.schedule_jump_params)

            time_pairs = list(zip(times[:-1], times[1:]))
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            counter = 0

            noise = th.randn_like(image_after_step)

            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = th.tensor([t_last] * shape[0],  # pylint: disable=not-callable
                                     device=device)
                # print(t_cur, t_last)
                if t_cur < t_last:  # reverse
                    counter += 1
                    with th.no_grad():
                        image_before_step = image_after_step.clone()
                        out = self.p_sample(
                            model,
                            image_after_step,
                            t_last_t,
                            counter=counter,
                            image_id=image_id,
                            fix_noise=noise,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                            conf=conf,
                            pred_xstart=pred_xstart
                        )
                        last_eps = out['eps']
                        last_log_variance = out['log_variance']
                        ##Repaint中的做法：直接通过xt得到xt-1
                        image_after_step = out["sample"]
                        ################################# 前向过程的xt-1也通过对predx0加噪得到
                        # alphas_comprod = _extract_into_tensor(self.alphas_cumprod, t_last_t, image_after_step.shape)
                        # image_after_step = th.sqrt(alphas_comprod) * out["pred_xstart"] + th.sqrt(1 - alphas_comprod) *\
                        #     th.randn_like(image_after_step)
                        #################################################################
                        pred_xstart = out["pred_xstart"]

                        sample_idxs[t_cur] += 1

                        yield out
                        # 观测每一步的图像和对应的预测x0
                        directory = conf.pget('data.eval.paper_face_mask.paths.reverse_processing') + '/' + str(
                            image_id)
                        make_dirs(directory)

                        # subfold = 'withp' if conf.pget('withp') else 'withoutp'
                        img = out["sample"]
                        img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
                        img = img.permute(0, 2, 3, 1)
                        img = img.contiguous().squeeze()
                        img = img.cpu().numpy()
                        img = Image.fromarray(img, mode='RGB')
                        full_p1 = os.path.join(directory, 'img' + '_' + str(counter).zfill(6) + '.jpg')
                        img.save(full_p1)

                        tmp_pred = out["pred_xstart"]
                        tmp_pred = ((tmp_pred + 1) * 127.5).clamp(0, 255).to(th.uint8)
                        tmp_pred = tmp_pred.permute(0, 2, 3, 1)
                        tmp_pred = tmp_pred.contiguous().squeeze()
                        tmp_pred = tmp_pred.cpu().numpy()
                        tmp_pred = Image.fromarray(tmp_pred, mode='RGB')
                        full_p2 = os.path.join(directory, 'pre' + '_' + str(counter).zfill(6) + '.jpg')
                        tmp_pred.save(full_p2)
                        #######################

                else:
                    counter += t_cur - t_last
                    t_shift = t_cur - t_last
                    image_before_step = image_after_step.clone()
                    image_after_step = self.undo(
                        image_before_step, image_after_step,
                        est_x_0=out['pred_xstart'], t=t_last_t + t_shift, model_kwargs=model_kwargs, debug=False,
                        conf=conf, eps=last_eps, last_log_variance=last_log_variance, jump=t_shift)
                    pred_xstart = out["pred_xstart"]


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def to_tensor(x, device):
    return th.from_numpy(x).to(device).float()
