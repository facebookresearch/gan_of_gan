# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import logging
import os
from collections import OrderedDict

import dlib
import imageio
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from face_align.face_alignment_ref import (
    image_align,
)
from face_align.landmarks_detector_dlib import (
    LandmarksDetector_dlib,
)
from losses import VGGLoss
from models_all import G_synthesis, G_mapping

syn_pt_path = "./karras2019stylegan-ffhq-1024x1024.for_g_synthesis.pt"
dlib_lm_path = "./shape_predictor_68_face_landmarks.dat"


def crop_and_rotate_img(opt, src_img, landmarks_det_68, img_name):
    """Crop and rotate the image based on facial landmarks, with post-processing
    to scale it to [-1, 1] and add extra axis on first dimension.

    Parameters
    ----------
    src_img : HXWXC numpy array.

    Returns
    -------
    numpy array
        1XCXHXW numpy array scaled to [-1, 1].

    """
    face_landmark = landmarks_det_68.get_landmarks(src_img)
    if face_landmark is None:
        logging.info("no landmark detected for {}".format(img_name))
        return None
    rotated_img_256 = image_align(src_img, face_landmark, output_size=256)

    imageio.imsave(
        "{}/{}_rotated.png".format(opt.output_folder, img_name), rotated_img_256
    )
    # scale to [-1, 1] and reshape to 1xcxhxw
    rotated_img_256 = (rotated_img_256 - 127.5) / 127.5
    rotated_img_256 = np.rollaxis(rotated_img_256, 2, 0)
    rotated_img_256 = np.expand_dims(rotated_img_256, 0)
    return rotated_img_256


def find_latent_from_images(opt, img_batch, generator):
    """Find the latent code from a batch of images using iterative backpropagation."""
    loss_l1 = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss()
    loss = VGGLoss().cuda()

    cur_latents = Variable(torch.zeros(1, 18, 512).cuda())
    cur_latents.requires_grad = True
    optZ = torch.optim.SGD([cur_latents], lr=1)

    for iter in range(opt.num_iterations):
        generated = generator.forward(cur_latents)
        # We need to downsample the output image from 1024x1024 to 256x256.
        # Since we use 256 x 256 images to compute the VGG losses.
        generated = F.upsample(generated, size=(256, 256), mode="bilinear")
        generated.clamp(-1, 1)

        if iter % 100 == 0:
            res_img = generated.detach().cpu().float().numpy()
            # reshape from batchxcxhxw to batchxhxwxc and scale to [0, 255].
            res_img = (np.transpose(res_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

            if opt.verbose:
                imageio.imsave(
                    "{}/regenerated_before_opt_{}.png".format(opt.output_folder, iter),
                    res_img[0],
                )

        recLoss = loss(generated, img_batch)
        recLoss_l1 = loss_l1(generated, img_batch)
        recLoss_l2 = loss_l2(generated, img_batch)
        if opt.loss_type == 1:
            total_loss = recLoss + recLoss_l1 * 5
        elif opt.loss_type == 2:
            total_loss = recLoss_l1
        elif opt.loss_type == 3:
            total_loss = recLoss_l2
        elif opt.loss_type == 4:
            total_loss = recLoss

        optZ.zero_grad()
        total_loss.backward(retain_graph=True)
        optZ.step()
        if iter % 100 == 0:
            logging.info(
                "iter: {}. vgg_loss: {:05f}, l1_loss: {:05f}, recloss: {:05f}".format(
                    iter,
                    recLoss.data.item(),
                    recLoss_l1.data.item(),
                    total_loss.data.item(),
                )
            )
    return cur_latents


def reinitialize_requires_grad_diff_layers(generator, opt):
    for name, param in generator.named_parameters():
        if "1024x1024" in name and opt.finetune_layers >= 1:
            param.requires_grad = True
        elif "512x512" in name and opt.finetune_layers >= 2:
            param.requires_grad = True
        elif "256x256" in name and opt.finetune_layers >= 3:
            param.requires_grad = True
        elif "128x128" in name and opt.finetune_layers >= 4:
            param.requires_grad = True
        elif "64x64" in name and opt.finetune_layers >= 5:
            param.requires_grad = True
        elif "32x32" in name and opt.finetune_layers >= 6:
            param.requires_grad = True
        elif "16x16" in name and opt.finetune_layers >= 7:
            param.requires_grad = True
        elif "8x8" in name and opt.finetune_layers >= 8:
            param.requires_grad = True
        elif "4x4" in name and opt.finetune_layers >= 9:
            param.requires_grad = True
        else:
            param.requires_grad = False


def finetune_weights_from_images(opt, img_batch, generator, latents):
    latents = latents.detach()
    loss = VGGLoss().cuda()
    loss_l1 = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss()
    optZ = torch.optim.SGD(generator.parameters(), lr=1)

    for iter in range(opt.num_iterations):
        total_recLoss = 0

        generated = generator.forward(latents)
        # We need to downsample the output image from 1024x1024 to 256x256.
        # Since we use 256 x 256 images to compute the VGG losses.
        generated = F.upsample(generated, size=(256, 256), mode="bilinear")
        generated.clamp(-1, 1)

        if iter % 100 == 0:
            res_img = generated.detach().cpu().float().numpy()
            # reshape from batchxcxhxw to batchxhxwxc and scale to [0, 255].
            res_img = (np.transpose(res_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

            if opt.verbose:
                imageio.imsave(
                    "{}/regenerated_{}.png".format(opt.output_folder, iter), res_img[0]
                )

        recLoss = loss(generated, img_batch)
        recLoss_l1 = loss_l1(generated, img_batch) * 5
        recLoss_l2 = loss_l2(generated, img_batch)
        if opt.loss_type == 1:
            total_loss = recLoss + recLoss_l1 * 5
        elif opt.loss_type == 2:
            total_loss = recLoss_l1
        elif opt.loss_type == 3:
            total_loss = recLoss_l2
        elif opt.loss_type == 4:
            total_loss = recLoss

        if iter % 100 == 0:
            logging.info(
                "recLoss: {} recLoss_l1: {:05f}".format(
                    recLoss.data.item(), recLoss_l1.data.item()
                )
            )
        total_loss = recLoss + recLoss_l1

        optZ.zero_grad()
        total_loss.backward(retain_graph=True)
        optZ.step()
        total_recLoss += total_loss.data.item()

        if iter % 100 == 0:
            logging.info("iter: {} loss: {:05f}".format(iter, total_recLoss))
    return generator


def regenerate_img(opt, img_batch, generator, exp_img_batch=None, img_names=None):
    """regenerate the image using stylegan.

    Parameters
    ----------
    img_batch : numpy array
            batchsize x height x width x channels.
    exp_img_batch: numpy array
            batchsize x height x width x channels. This is optional.

    Returns:
    res_img: numpy array.
        scaled to [0, 255] and cast as uint8.
    """
    latents = find_latent_from_images(opt, img_batch, generator)

    cur_latents = latents
    with torch.no_grad():
        res_img = generator.forward(cur_latents)
        res_img.clamp(-1, 1)

    res_img = res_img.cpu().float().numpy()
    # reshape from batchxcxhxw to batchxhxwxc and scale to [0, 255].
    res_img = (np.transpose(res_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

    if opt.verbose:
        imageio.imsave(
            "{}/{}_regenerated_before_opt.png".format(opt.output_folder, img_names[0]),
            res_img[0],
        )

    # We then fix the latents and finetune weights from images
    # Set the requires grad for each layer
    reinitialize_requires_grad_diff_layers(generator, opt)
    finetune_weights_from_images(opt, img_batch, generator, latents)

    latents_numpy = latents.data.cpu().numpy()
    if opt.verbose:
        # save the latent code
        outfile = "{}/{}_latent.npy".format(opt.output_folder, img_names[0])
        np.save(outfile, latents_numpy)
        # save the fine-tuned model as well:
        torch.save(
            generator.state_dict(),
            "{}/karras2019stylegan-ffhq-1024x1024.for_g_synthesis_finetuned.pt".format(
                opt.output_folder
            ),  # noqa
        )

    # save the image
    with torch.no_grad():
        res_img = generator.forward(cur_latents)
        res_img.clamp(-1, 1)

    res_img = res_img.cpu().float().numpy()
    # reshape from batchxcxhxw to batchxhxwxc and scale to [0, 255].
    res_img = (np.transpose(res_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

    imageio.imsave(
        "{}/{}_regenerated.png".format(opt.output_folder, img_names[0]), res_img[0]
    )
    return generator, latents_numpy


def read_image_to_numpy(img_name):
    """Read image into numpy array. The image could be numpy for celebA-HQ where
    the shape is 1x3x1024x1024."""
    ext = img_name.split(".")[-1]
    if ext == "npy":
        img_np = np.load(img_name)
        if len(img_np.shape) > 3:
            img_np = np.squeeze(img_np)
        if img_np.shape[0] == 3 or img_np.shape[0] == 1:
            img_np = np.rollaxis(img_np, 0, 3)
        return img_np
    else:
        return imageio.imread(img_name)


def generate_fake_images(opt, generator, landmarks_det_68):
    """Generate fake images from real images. We read the images line by line until
    it forms a batch of images, which is given to the trained StyeGAN model to infer
    the latent code and regenerate the image. It is then post-processed to blend the
    face with the original image."""
    # ----------------------------------------------------------------------
    # Rotate and rectify images of the given batch.
    # ----------------------------------------------------------------------
    rotated_img_256_batch = []
    img_names = []
    # for src_img, img_name in zip(src_imgs, img_names):
    src_img = read_image_to_numpy(opt.img_name)

    img_name_no_extension = ".".join(os.path.basename(opt.img_name).split(".")[:-1])
    img_names.append(img_name_no_extension)
    if opt.no_crop_and_rotate:
        rotated_img_256 = scipy.misc.imresize(src_img, (256, 256))
        imageio.imsave(
            "{}/{}_rotated.png".format(opt.output_folder, img_name_no_extension),
            rotated_img_256,
        )
        rotated_img_256 = (rotated_img_256 - 127.5) / 127.5
        rotated_img_256 = np.rollaxis(rotated_img_256, 2, 0)
        rotated_img_256 = np.expand_dims(rotated_img_256, 0)
    else:
        rotated_img_256 = crop_and_rotate_img(
            opt, src_img, landmarks_det_68, img_name_no_extension
        )
    if rotated_img_256 is None:
        logging.info('No face detected.')
        exit()
    rotated_img_256_batch.append(rotated_img_256)

    # ----------------------------------------------------------------------
    # Infer the latent style and regenerate the image.
    # ----------------------------------------------------------------------
    rotated_img_256_batch = np.vstack(rotated_img_256_batch)
    rotated_img_256_batch = Variable(
        torch.from_numpy(rotated_img_256_batch).float().cuda()
    )

    generator, latents_numpy = regenerate_img(
        opt, rotated_img_256_batch, generator, exp_img_batch=None, img_names=img_names
    )  # generated images are 1024 x 1024.
    return generator, latents_numpy


def initialize_models_all(opt):
    """Initialize the model and load the pre-trained weights. Note that we only
    build the synthesis network which takes styles as input (no mapping network
    is necessary for our task)."""
    # load mapping network
    g_all = nn.Sequential(
        OrderedDict(
            [
                ("g_mapping", G_mapping()),
                # ('truncation', Truncation(avg_latent)),
                (
                    "g_synthesis",
                    G_synthesis(
                        randomize_noise=False,
                        resolution=opt.resolution,
                        use_random_initial_noise=opt.use_random_initial_noise,
                    ),
                ),
            ]
        )
    )
    g_all.load_state_dict(
        torch.load(
            "./karras2019stylegan-ffhq-1024x1024.for_g_all.pt"  # noqa
        )
    )
    state = g_all.g_synthesis.state_dict()
    state.update(torch.load(syn_pt_path))
    loaded_dict = {k: state[k] for k in g_all.g_synthesis.state_dict()}
    g_all.g_synthesis.load_state_dict(loaded_dict)
    g_all.eval()
    g_mapping = g_all.g_mapping.cuda()
    g_synthesis = g_all.g_synthesis.cuda()
    return g_mapping, g_synthesis


def generate_new_imgs(opt, g_mapping, g_synthesis, latents_numpy):
    """Generate new images based on optimized model and style mixing."""
    # load latent numpy
    latents = Variable(torch.from_numpy(latents_numpy).cuda())
    if not opt.randomize_seed:
        torch.manual_seed(20)
    for i in range(0, opt.how_many_samples):
        with torch.no_grad():
            rand_z = torch.randn(1, 512).cuda()
            rand_latents = g_mapping.forward(rand_z)

            latents[:, : opt.how_many_layers, :] = rand_latents[
                                                   :, : opt.how_many_layers, :
                                                   ]
            imgs = g_synthesis.forward(latents)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0  # normalization to 0..1 range

        res_img = imgs.cpu().float().numpy()
        # reshape from batchxcxhxw to batchxhxwxc and scale to [0, 255].
        res_img = (np.transpose(res_img, (0, 2, 3, 1)) + 1) / 2.0 * 255.0

        logging.info(
            "generating {0}/{1:06d}_synthesized.jpg".format(opt.output_folder, i)
        )
        imageio.imsave(
            "{0}/{1:06d}_synthesized.jpg".format(opt.output_folder, i), res_img[0]
        )


# noinspection DuplicatedCode
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN deepfake.")
    parser.add_argument(
        "--img_name",
        metavar="N",
        type=str,
        default="",  # noqa 501
        help="The path to the image list.",
    )
    parser.add_argument(
        "--resolution",
        metavar="N",
        type=int,
        default=1024,  # noqa 501
        help="Model resolution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If specified, save intermediate results.",
    )
    parser.add_argument(
        "--exp_name",
        metavar="N",
        type=str,
        default="test",  # noqa 501
        help="The experiment name.",
    )
    parser.add_argument(
        "--output_folder",
        metavar="N",
        type=str,
        default="./test_results",  # noqa 501
        help="The experiment name.",
    )
    parser.add_argument(
        "--num_iterations",
        metavar="N",
        type=int,
        default=1000,
        help="Number of iterations for optimization.",
    )
    parser.add_argument(
        "--finetune_layers",
        metavar="N",
        type=int,
        default=9,
        help="Number of layers for optimization.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
    )
    parser.add_argument(
        "--no_crop_and_rotate",
        action="store_true",
        help="If specified, no cropping.",  # noqa 501
    )
    parser.add_argument(
        "--use_random_initial_noise",
        action="store_true",
        help="If specified, use random noise instead of zero noise for stylegan.",  # noqa 501
    )
    parser.add_argument(
        "--randomize_seed",
        action="store_true",
        help="Whether to randomize the image generation.",  # noqa 501
    )
    parser.add_argument(
        "--loss_type",
        metavar="N",
        type=int,
        default=1,
        help="Different losses. 1: VGG + 5* L1 2. L1, 3. L2, 4. VGG.",
    )
    parser.add_argument(
        "--how_many_layers",
        metavar="N",
        type=int,
        default=15,
        help="How many initial layers that we use random styles.",
    )
    parser.add_argument(
        "--how_many_samples",
        metavar="N",
        type=int,
        default=10,
        help="How many samples to generate.",
    )
    opt = parser.parse_args()

    opt.output_folder = os.path.join(opt.output_folder, opt.exp_name)
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    # initialize gpu
    str_ids = opt.gpu_ids.split(",")
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    logging.info("initializing model...")
    g_mapping, generator = initialize_models_all(opt)

    logging.info("initializing landmark detector...")
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(dlib_lm_path)

    landmarks_det_68 = LandmarksDetector_dlib(
        face_detector, landmark_detector
    )

    logging.info("processing images...")

    g_synthesis, latents_numpy = generate_fake_images(
        opt, generator, landmarks_det_68
    )
    generate_new_imgs(opt, g_mapping, g_synthesis, latents_numpy)
