# -*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

from fdl.federation import Federation
from fdl.tensorflow.vfl_graph import FLGuestGraph, FLHostGraph
from fdl.tensorflow.comm_ops import send_op, recv_op
from fdl.tensorflow.privacy.proj_pert import projection_perturb_fn
from fdl.tensorflow.privacy.marvell import marvell_perturb_fn
from fdl.tensorflow.privacy.label_protection import gaussian_perturb_fn
from fdl.tensorflow.privacy.label_protection import maxnorm_gaussian_perturb_fn
from fdl.tensorflow.privacy.label_protection import differential_private_perturb_fn
from fdl.tensorflow.privacy.label_inference import label_infer_from_norm
from fdl.tensorflow.privacy.label_inference import label_infer_from_direction

from model.model import DIN,BST
import os, sys, traceback
import time
import logging
from collections import defaultdict

def define_model(args):
    guest_user_ids = tf.placeholder(tf.int32, [None, 1], name="guest-user-ids")
    guest_item_ids = tf.placeholder(tf.int32, [None, 1], name="guest-item-ids")
    guest_timestamps = tf.placeholder(tf.float32, [None, 1], name="guest-timestamps")
    guest_behavior_timestamps = tf.placeholder(tf.float32, [None, args.history_size], name="guest_behavior_timestamps")
    labels = tf.placeholder(tf.int32, [None, 1], name="labels")
    if args.data == "movielens_1m":
        host_sparse = tf.placeholder(tf.int32, [None, 4], name="host-sparse")  # user feats, drop the id column
        host_dense = None
        guest_sparse = tf.placeholder(tf.int32, [None, 3], name="guest-sparse")  # item feats, drop the id column
        guest_dense = None
        guest_behaviors_sparse = tf.placeholder(tf.int32, [None, args.history_size, 3], name="guest-behaviors-sparse")
        guest_behaviors_dense  = None
        guest_time_diff = tf.placeholder(tf.float32, [None, args.history_size], name="guest-time-diff")

    elif args.data == "ad_click":
        host_sparse = tf.placeholder(tf.int32, [None, 8], name="host-sparse")
        host_dense = None
        guest_sparse = tf.placeholder(tf.int32, [None, 4], name="guest-sparse")
        guest_dense = tf.placeholder(tf.float32, [None, 1], name="guest-dense")
        guest_behaviors_sparse = tf.placeholder(tf.int32, [None, args.history_size, 4], name="guest-behaviors-sparse")
        guest_behaviors_dense  = tf.placeholder(tf.float32, [None, args.history_size, 1] , name="guest-behaviors-dense")
        guest_time_diff = tf.placeholder(tf.float32, [None, args.history_size], name="guest-time-diff")

    elif args.data == "kuairec":
        host_sparse = tf.placeholder(tf.int32, [None, 22], name="host-sparse")
        host_dense = tf.placeholder(tf.float32, [None, 4], name="host-sparse")
        guest_sparse = tf.placeholder(tf.int32, [None, 14], name="guest-sparse")
        guest_dense = tf.placeholder(tf.float32, [None, 47], name="guest-dense")
        guest_behaviors_sparse = tf.placeholder(tf.int32, [None, args.history_size, 14], name="guest-behaviors-sparse")
        guest_behaviors_dense  = tf.placeholder(tf.float32, [None, args.history_size, 47] , name="guest-behaviors-dense")
        guest_time_diff = tf.placeholder(tf.float32, [None, args.history_size], name="guest-time-diff")
    else:
        raise ValueError(f"No such data: {args.data}")

    if args.model == "DIN":
        model = DIN(guest_bottom_mlp_units = [256,128,96,64], host_bottom_mlp_units = [256,128,96,64], dropout_rate=0.5, l2_reg=0.01)
        optimizer = tf.train.AdagradOptimizer(args.eta)

    elif args.model == "BST":
        model = BST(guest_embedding_size=256, guest_num_buckets=100000,
                    host_embedding_size=256, host_num_buckets=100000,
                    num_transformer_layers=1, att_head_num=8, dropout_rate=0.2, transformer_seed= np.random.randint(0, 1000))
        optimizer = tf.train.AdagradOptimizer(args.eta)

    else:
        raise ValueError(f"No such model: {args.model}")

    if args.party == "guest":
        graph = FLGuestGraph(deps=[guest_sparse])
        host_act = graph.remote_bottom("HostAct", shape=[None, 64])
        with tf.device(args.device):
            num_u2i = tf.placeholder(tf.int32, [None], name="num-items")
            pesudo_labels = tf.placeholder(tf.int32, [None], name="pesudo-labels")
            repeated_host_act = tf.repeat(host_act, num_u2i, axis=0)

        top_model = model.item_model(labels, repeated_host_act, guest_behavior_timestamps, guest_sparse, guest_behaviors_sparse, guest_behaviors_dense, guest_dense, guest_time_diff)

        # minimize, perturb and send derivatives
        perturb_fn = get_perturb_fn(args, pesudo_labels)
        train_op = graph.minimize(
            optimizer, top_model["loss"],
            perturb_fn=perturb_fn,
            return_grads=False)
        pred_op = graph.predict(top_model["logits"])
        return (guest_sparse,
                guest_dense,
                num_u2i,
                guest_behaviors_sparse, guest_behaviors_dense,
                guest_user_ids, guest_item_ids,
                guest_timestamps, guest_behavior_timestamps, guest_time_diff, labels,
                pesudo_labels, top_model, train_op, pred_op)
    else:
        graph = FLHostGraph()
        # define bottom model of host
        host_act = model.user_model(host_sparse, host_dense)
        # send the output of bottom model of host
        graph.send_bottom("HostAct", host_act)
        # receive derivatives, minimize
        train_op, _, devs_and_acts = graph.minimize(optimizer, return_grads=True)
        pred_op = graph.predict()
        # infer labels via derivatives
        host_dev = list(filter(lambda x: x[1] is host_act, devs_and_acts))[0][0]
        infer_logits = []
        infer_fns = get_infer_fns()
        if len(infer_fns) > 0:
            for infer_name, infer_fn in infer_fns:
                infer_logits.append((infer_name, infer_fn(host_dev)))
        return host_sparse, host_dense, train_op, pred_op, infer_logits, host_act, devs_and_acts


def get_infer_fns():
    infer_fns = [
        ("Norm", label_infer_from_norm),
        ("Direction", label_infer_from_direction)
    ]
    return infer_fns


def get_perturb_fn(args, labels):
    if not args.perturb or args.perturb == "none":
        return None
    elif args.perturb == "iso-proj":
        return lambda dev: projection_perturb_fn(
            dev, labels, sum_kl_bound=args.sum_kl_bound, iso_proj=True)
    elif args.perturb == "proj":
        return lambda dev: projection_perturb_fn(
            dev, labels, sum_kl_bound=args.sum_kl_bound, iso_proj=False)
    elif args.perturb == "marvell":
        return lambda dev: marvell_perturb_fn(
            dev, labels, init_scale=args.init_scale)
    elif args.perturb == "gaussian":
        return lambda dev: gaussian_perturb_fn(
            dev, labels, scale=args.init_scale)
    elif args.perturb == "maxnorm_gaussian":
        return lambda dev: maxnorm_gaussian_perturb_fn(
            dev, labels, scale=args.init_scale)
    elif args.perturb == "laplace_dp":
        return lambda dev: differential_private_perturb_fn(
            dev, labels, eps=args.dp_eps, dist="laplace")
    elif args.perturb == "bernouli_dp":
        return lambda dev: differential_private_perturb_fn(
            dev, labels, eps=args.dp_eps, dist="bernouli")
    else:
        raise ValueError(f"No such perturbance method: {args.perturb}")