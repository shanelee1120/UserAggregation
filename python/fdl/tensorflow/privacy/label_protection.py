#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np
import logging

# from fdl.tensorflow.privacy.proj_pert import projection_perturb_fn
# from fdl.tensorflow.marvell import marvell_perturb_fn

def gaussian_perturb_fn(x, y, scale):
    if scale <= 0:
        raise ValueError("Scale should be positive")
    logging.info(f"scale[{scale}]")
    
    if len(y.shape) == 2 and (y.shape[0] == 1 or y.shape[1] == 1):
        y = tf.reshape(y, [-1])
    elif len(y.shape) != 1:
        raise ValueError(f"Unsupported shape for y: {y.shape}")
    if y.dtype != tf.float32:
        y = tf.cast(y, tf.float32)
    
    x_shape = tf.shape(x)
    pos_indices = tf.where(y)
    neg_indices = tf.where(1 - y)
    pos_x = tf.gather_nd(x, pos_indices)
    neg_x = tf.gather_nd(x, neg_indices)
    pos_mean = tf.reduce_mean(pos_x, axis=0)
    neg_mean = tf.reduce_mean(neg_x, axis=0)
    mean_diff = pos_mean - neg_mean
    mean_diff_norm = tf.math.sqrt(tf.reduce_sum((pos_mean - neg_mean) ** 2))

    noise_mean = tf.zeros([x_shape[1]])
    noise_std = tf.ones([x_shape[1]]) * (tf.math.sqrt(scale) * mean_diff_norm)
    noise_dist = tf.distributions.Normal(noise_mean, noise_std)

    noise = noise_dist.sample([x_shape[0]])
    perturbed_x = x + noise
    return perturbed_x


def maxnorm_gaussian_perturb_fn(x, y, scale):
    if scale <= 0:
        raise ValueError("Scale should be positive")
    logging.info(f"scale[{scale}]")
    
    x_shape = tf.shape(x)
    scale_sqrt = tf.math.sqrt(scale / tf.cast(x_shape[1], dtype=tf.float32))
    max_norm = tf.reduce_max(tf.norm(x, axis=1))

    noise_mean = tf.zeros([x_shape[1]])
    noise_std = tf.ones([x_shape[1]]) * (scale_sqrt * max_norm)
    noise_dist = tf.distributions.Normal(noise_mean, noise_std)

    noise = noise_dist.sample([x_shape[0]])
    perturbed_x = x + noise
    return perturbed_x


def differential_private_perturb_fn(x, y, eps=1.0, dist="laplace"):
    """
    Differentially Private Label Protection in Split Learning. https://arxiv.org/abs/2203.02073
    """

    if eps < 0:
        raise ValueError(f"Epsilon should be non-negative")
    if dist not in ("laplace", "bernouli"):
        raise ValueError(f"Unknown distribution: {dist}")
    
    if dist == "laplace":
        b = 1.0 / eps
        logging.info(f"eps[{eps}] dist[Laplace({b})]")
    elif dist == "bernouli":
        p = 1.0 / (np.exp(eps) + 1.0)
        logging.info(f"eps[{eps}] dist[Bernouli({p})]")
    
    if len(y.shape) == 2 and (y.shape[0] == 1 or y.shape[1] == 1):
        y = tf.reshape(y, [-1])
    elif len(y.shape) != 1:
        raise ValueError(f"Unsupported shape for y: {y.shape}")
    if y.dtype != tf.float32:
        y = tf.cast(y, tf.float32)
    

    x_shape = tf.shape(x)
    pos_indices = tf.where(y)
    neg_indices = tf.where(1 - y)
    pos_x = tf.gather_nd(x, pos_indices)
    neg_x = tf.gather_nd(x, neg_indices)
    pos_mean = tf.reduce_mean(pos_x, axis=0)
    neg_mean = tf.reduce_mean(neg_x, axis=0)
    mean_diff = pos_mean - neg_mean

    if dist == "laplace":
        noise_dist = tf.distributions.Laplace(0.0, b)
    elif dist == "bernouli":
        noise_dist = tf.distributions.Bernoulli(probs=p, dtype=tf.float32)
    
    noise = noise_dist.sample([x_shape[0]])
    sign = -2 * y + 1 # y=1 --> -1, y=0 --> 1
    noise = tf.einsum('i,j->ij', noise * sign, mean_diff)
    perturbed_x = x + noise

    # mean_diff = pos_dev - neg_dev
    # sign = -2 * y + 1 # y=1 --> -1, y=0 --> 1
    # u = noise_dist.sample([x_shape[0]])
    # noise = tf.reshape(u * sign, [x_shape[0], 1])
    # noise = noise * mean_diff
    # perturbed_x = x + noise

    return perturbed_x