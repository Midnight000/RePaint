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
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse

import torch
import torch as th
import torch.nn.functional as F
import time

from PIL import Image

import conf_mgt
from utils import yamlread, Dis_Transform, Parameter
from guided_diffusion import dist_util
import numpy as np

# Workaround
try:
    import ctypes

    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):
    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)
    print(dl.__len__())
    image_id = 0
    for batch in iter(dl):
        image_id = image_id + 1
        # if image_id <= 37:
        #     continue
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask
            weight_mask = gt_keep_mask
            weight_mask = (weight_mask * 255).to(th.uint8)
            weight_mask = weight_mask.squeeze().cpu().numpy()
            weight_mask = np.transpose(weight_mask, (1, 2, 0))
            weight_mask = Image.fromarray(weight_mask, 'RGB')
            weight_mask = Dis_Transform.dis_transform(weight_mask)
            total = weight_mask.sum()/256/256/256/3
            # print(total)
            factor = 0.95
            tmpfactor = 0.90
            first = 0.3
            for i in range(18):
                if total < first:
                    factor = tmpfactor
                tmpfactor = tmpfactor - 0.05
                total = total * 2
            print(factor)

            ####################################### dense
            maxm = np.max(weight_mask)
            weight_mask = (np.transpose(weight_mask, (2, 0, 1)) / maxm) ** 2
            weight_mask = weight_mask * (1 - 0.95) + 0.95
            weight_mask = torch.Tensor(weight_mask).to(device).unsqueeze(0)
            model_kwargs["weight_mask"] = weight_mask

            ####################################### fixed_eta
            # maxm = np.max(weight_mask)
            # weight_mask = (np.transpose(weight_mask, (2, 0, 1)) / maxm) ** 2
            # weight_mask = torch.Tensor(weight_mask).to(device).unsqueeze(0)
            # eta = 0.01 * int(conf["eta"])
            # model_kwargs["weight_mask"] = th.ones_like(weight_mask) * eta

            ####################################### fixed_eta+pixel_position
            # maxm = np.max(weight_mask)
            # weight_mask = (np.transpose(weight_mask, (2, 0, 1)) / maxm) ** 2
            # eta = 0.01 * int(conf["eta"])
            # weight_mask = weight_mask * (1 - eta) + eta
            # weight_mask = torch.Tensor(weight_mask).to(device).unsqueeze(0)
            # model_kwargs["weight_mask"] = weight_mask

            ####################################### fun_test
            # maxm = np.max(weight_mask)
            # weight_mask = (np.transpose(weight_mask, (2, 0, 1)))
            # # weight_mask = np.random.random_integers(0, 100, weight_mask.shape)
            # weight_mask = np.zeros_like(weight_mask)
            # weight_mask[:, ::2, ::2] = 1
            # weight_mask[:, 1::2, 1::2] = 1
            # weight_mask = torch.Tensor(weight_mask).to(device).unsqueeze(0)
            # weight_mask = th.where(weight_mask % 2 == 0, 0.3, 1.0) * 1.0

            model_kwargs["weight_mask"] = weight_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf,
            image_id=image_id
        )
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--withp', action='store_true', default=False)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    conf_arg.update({'withp': args.get('withp')})

    # for x in range(0, 256-128+1, 256):
    #     for y in range(128, 256-64+1, 32):
    #         conf_arg = Parameter.change(x, y, 64, 256, conf_arg)
    #         print(conf_arg)
    main(conf_arg)
