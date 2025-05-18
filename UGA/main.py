# -*- coding:utf-8 -*-

import sys
import os

import tensorflow.compat.v1 as tf
# import tensorflow as tf
import numpy as np
from sklearn import metrics

from fdl.federation import Federation
from utils import define_model
from dataset import load_data

import os, sys, traceback
import time
import random
import argparse
import threading
import logging


# logging.basicConfig(level=logging.INFO)

def handle_invalid_values(tensor):
    tensor = np.nan_to_num(tensor, nan=0.0)
    tensor = np.where(np.isinf(tensor), 1e308, tensor)
    tensor = np.clip(tensor, -1e308, 1e308)
    return tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, default="test_task",
                        help="Task id")
    parser.add_argument("--party", type=str, default="guest",
                        choices=("guest", "host"), help="Party name")
    parser.add_argument("--config-file", type=str, default="config.yaml",
                        help="Config yaml")
    parser.add_argument("--data", type=str,
                        choices=("movielens_1m", "book_crossing", "ali_ccp", "ad_click", "digix_video", "kuairec", "tenrec_article"),
                        help="Name of dataset")
    parser.add_argument("--device", type=str, default="/gpu:0",
                        help="Name of device")
    parser.add_argument("--max-updates", type=int, default=1000000,
                        help="Max number of updates")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--pack-size", type=int, default=16,
                        help="Packing size")
    parser.add_argument("--valid-freq", type=int, default=10000,
                        help="Frequency of evaluation on valid set")
    parser.add_argument("--print-freq", type=int, default=1,
                        help="Frequency of printing metrics")
    parser.add_argument("--perturb", type=str, default="iso-proj", 
                        choices=("proj", "iso-proj", "marvell", "gaussian", "maxnorm_gaussian", 
                                 "laplace_dp", "bernouli_dp", "mixpro", "none"), 
                        help="Perturbation method")
    parser.add_argument("--sum-kl-bound", type=float, default=4.0,
                        help="Upper bound of sum KL divergence (for proj and iso-proj")
    parser.add_argument("--init-scale", type=float, default=4.0,
                        help="Initial value of P is scale * g (for marvell)")
    parser.add_argument("--dp-eps", type=float, default=1.0, 
                        help="Epsilon for differential privacy")
    parser.add_argument("--history-size", type=int, default=5,
                        help="Size of user behavior history")
    parser.add_argument("--model", type=str, default="DIN",
                        choices=("DIN", "BST"),
                        help="Model name")
    parser.add_argument("--pseudo-labels", type=str, default="conservative",
                        choices=("conservative", "random", "all", "vote"),
                        help="Pseudo labels generation method")

    args = parser.parse_args()
    assert args.batch_size % args.pack_size == 0
    logging.info(f"Args: {args}")
    return args


def main(args):
    # define model
    model = define_model(args)

    if args.party == "guest":
        (guest_sparse,
         guest_dense,
         num_u2i,
         guest_behaviors_sparse, guest_behaviors_dense,
         guest_user_ids, guest_item_ids,
         guest_timestamps, guest_behavior_timestamps, guest_time_diff, labels,
         pesudo_labels, top_model, train_op, pred_op) = model
    else:
        host_sparse, host_dense, train_op, pred_op, infer_logits, host_act, devs_and_acts= model
        infer_names = [t[0] for t in infer_logits]
        infer_fetches = [t[1] for t in infer_logits]

    # init session
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init_op)

    # load data
    dataset = load_data(args)

    Federation.sync()
    logging.info("Start training...")
    train_start = time.time()

    def get_time_diff(guest_timestamps, guest_behavior_timestamps):
        time_diff = guest_timestamps.reshape(-1, 1) - guest_behavior_timestamps
        return time_diff

    def valid_fn(num_updates):
        valid_start = time.time()
        num_batch = 0
        if args.party == "guest":
            all_logits, all_labels = [], []
            batch_loss_list, batch_acc_list = [], []
        for batch in dataset.valid_iterator():
            if args.party == "guest":
                (batch_item_sparse_feats, batch_item_dense_feats,
                 batch_num_u2i,
                 # batch_user_ids,
                 batch_item_ids,
                 batch_behaviors_sparse, batch_behaviors_dense,
                 batch_timestamps, batch_behavior_timestamps,
                 batch_labels) = batch
                time_diff = get_time_diff(batch_timestamps, batch_behavior_timestamps)
                feed_dict = {
                    num_u2i: batch_num_u2i,
                    # guest_user_ids: batch_user_ids.reshape(-1, 1),
                    guest_item_ids: batch_item_ids.reshape(-1, 1),
                    guest_behaviors_sparse: batch_behaviors_sparse,
                    guest_timestamps: batch_timestamps.reshape(-1, 1),
                    guest_behavior_timestamps: batch_behavior_timestamps,
                    guest_time_diff: time_diff,
                    labels: batch_labels.reshape(-1, 1),
                }
                if guest_sparse is not None:
                    feed_dict[guest_sparse] = batch_item_sparse_feats
                if guest_dense is not None:
                    feed_dict[guest_dense] = batch_item_dense_feats
                if guest_behaviors_dense is not None:
                    feed_dict[guest_behaviors_dense] = batch_behaviors_dense
                loss_val, acc_val, logits_val = sess.run(
                    [top_model["loss"], top_model["acc"], pred_op],
                feed_dict = feed_dict)
                loss_val = handle_invalid_values(loss_val)
                acc_val = handle_invalid_values(acc_val)
                logits_val = handle_invalid_values(logits_val)
                all_logits.append(logits_val.reshape(-1))
                all_labels.append(batch_labels.reshape(-1))
                batch_loss_list.append(loss_val)
                batch_acc_list.append(acc_val)
            else:
                host_sparse_feats, host_dense_feats = batch
                feed_dict = {host_sparse: host_sparse_feats}
                if host_dense_feats is not None:
                    feed_dict[host_dense] = host_dense_feats
                sess.run(pred_op, feed_dict)
            num_batch += 1

        if args.party == "guest":
            loss = np.average(batch_loss_list)
            acc = np.average(batch_acc_list)
            all_logits = np.concatenate(all_logits)
            all_labels = np.concatenate(all_labels)
            fpr, tpr, _ = metrics.roc_curve(all_labels, all_logits)
            auc = metrics.auc(fpr, tpr)
            Federation.sync()
            logging.info(
                f"Validation after {num_updates} updates "
                f"cost {time.time() - valid_start:.4f} seconds, "
                f"{time.time() - train_start:.4f} elapsed, "
                f"Loss[{loss:.4f}] Accuracy[{acc:.4f}] AUC[{auc:.4f}]")
        else:
            Federation.sync()
            logging.info(
                f"Validation after {num_updates} updates "
                f"cost {time.time() - valid_start:.4f} seconds, "
                f"{time.time() - train_start:.4f} elapsed")


    num_total_updates = 0
    train_iterator = dataset.train_iterator(args.max_updates)
    while num_total_updates < args.max_updates:
        batch_start = time.time()
        need_print = ((num_total_updates + 1) % args.print_freq == 0)

        if args.party == "guest":
            (batch_item_sparse_feats, batch_item_dense_feats,
             batch_pesudo_labels, batch_num_u2i,
             batch_item_ids,
             batch_behaviors_sparse, batch_behaviors_dense,
             batch_behavior_timestamps, batch_timestamps,
             batch_labels, pos_ratio) = next(train_iterator)

            time_diff = get_time_diff(batch_timestamps, batch_behavior_timestamps)

            def generate_pesudo_labels(batch_pesudo_labels, mode = "conservative"):
                if mode == "conservative": # ours
                    # if there is any positive item in batch_pesudo_labels, set all items to 1
                    batch_pesudo_labels[batch_pesudo_labels != 0] = 1
                    return batch_pesudo_labels
                if mode == "random":
                    # random sample pos_ratio * batch_size items from batch_pesudo_labels
                    pseudo_labels = np.zeros_like(batch_pesudo_labels)
                    pseudo_labels[np.random.choice(len(batch_pesudo_labels), int(pos_ratio * len(batch_pesudo_labels)), replace=False)] = 1
                    return pseudo_labels
                if mode == "all":
                    # if there is any negative item in batch_pesudo_labels, set all items to 0
                    batch_pesudo_labels[batch_pesudo_labels != args.pack_size] = 0
                    batch_pesudo_labels[batch_pesudo_labels == args.pack_size] = 1
                    return batch_pesudo_labels
                if mode == "vote":
                    # vote for the most frequent label in batch_pesudo_labels
                    batch_pesudo_labels[batch_pesudo_labels > args.pack_size // 2] = 1
                    batch_pesudo_labels[batch_pesudo_labels <= args.pack_size // 2] = 0
                    return batch_pesudo_labels

                
            batch_pesudo_labels = generate_pesudo_labels(batch_pesudo_labels, mode = args.pseudo_labels)

            feed_dict = {
                pesudo_labels: batch_pesudo_labels,
                num_u2i: batch_num_u2i,
                # guest_user_ids: batch_user_ids.reshape(-1, 1),
                guest_item_ids: batch_item_ids.reshape(-1, 1),
                guest_behaviors_sparse: batch_behaviors_sparse,
                guest_behavior_timestamps: batch_behavior_timestamps,
                guest_time_diff: time_diff,
                labels: batch_labels.reshape(-1, 1)
            }
            if guest_sparse is not None:
                feed_dict[guest_sparse]= batch_item_sparse_feats
            if guest_dense is not None:
                feed_dict[guest_dense]= batch_item_dense_feats
            if guest_behaviors_dense is not None:
                feed_dict[guest_behaviors_dense]= batch_behaviors_dense

            loss_val, acc_val, logits_val, _ = sess.run(
                [top_model["loss"], top_model["acc"], top_model["logits"], train_op],
                feed_dict = feed_dict
            )

            loss_val = handle_invalid_values(loss_val)
            acc_val = handle_invalid_values(acc_val)
            logits_val = handle_invalid_values(logits_val)
        else:
            host_sparse_feats, host_dense_feats, batch_labels, batch_num_u2i, pos_ratio = next(train_iterator)
            feed_dict = {host_sparse: host_sparse_feats}
            if host_dense_feats is not None:
                feed_dict[host_dense] = host_dense_feats

            if need_print:
                result = sess.run(
                    # [train_op, tf.shape(host_act),tf.shape(devs_and_acts), host_act, devs_and_acts, *infer_fetches], 
                    [train_op, *infer_fetches], 
                    feed_dict=feed_dict
                )
                # train_op_val, host_act_shape, devs_and_acts_shape, host_act_val, devs_and_acts_val, *infer_vals = result
                train_op_val, *infer_vals = result
            else:
                sess.run(pred_op, feed_dict=feed_dict)

        batch_cost = time.time() - batch_start
        num_total_updates += 1
        if need_print:
            if args.party == "guest":
                batch_labels = batch_labels.reshape(-1)
                fpr, tpr, _ = metrics.roc_curve(batch_labels, logits_val)
                auc = metrics.auc(fpr, tpr)
                logging.info(
                    f"Train Update[{num_total_updates}] "
                    f"Time[{batch_cost:.4f}] "
                    f"Loss[{loss_val:.4f}] "
                    f"Accuracy[{acc_val:.4f}] "
                    f"AUC[{auc:.4f}]")
            else:
                infer_msg = ""
                infer_aucs = []
                infer_accs = []
                for i in range(len(infer_vals)):
                    infer_val = handle_invalid_values(infer_vals[i])
                    batch_labels = batch_labels.reshape(-1)
                    infer_val = infer_val.reshape(-1)
                    if args.pack_size > 1:
                        infer_val = np.repeat(infer_val, batch_num_u2i)

                    fpr, tpr, _ = metrics.roc_curve(batch_labels, infer_val)
                    infer_auc = metrics.auc(fpr, tpr)
                    infer_auc = max(infer_auc, 1.0 - infer_auc)
                    infer_aucs.append(infer_auc)
                    
                    predict_labels = np.zeros_like(batch_labels)
                    predict_labels[np.argsort(infer_val)[-int(pos_ratio * args.batch_size):]] = 1
                    infer_acc = np.sum(predict_labels == batch_labels) / len(batch_labels)
                    infer_accs.append(infer_acc)

                infer_msg = " ".join(f"{t[0]}AUC[{t[1]:.4f}]" for t in zip(infer_names, infer_aucs))
                infer_msg += " " + " ".join(f"{t[0]}Acc[{t[1]:.4f}]" for t in zip(infer_names, infer_accs))
                
                # size of host_act_val and devs_and_acts_val in MB
                # host_act_size = host_act_val.nbytes / (1024 * 1024)
                # devs_and_acts_size = sum(np.array(a).nbytes for a in devs_and_acts_val) / (1024 * 1024)
                # total_size = host_act_size + devs_and_acts_size
                logging.info(
                    f"Train Update[{num_total_updates}] "
                    f"Time[{batch_cost:.4f}] "
                    # f"host_act_val shape: {host_act_shape} "
                    # f"devs_and_acts_val shape: {devs_and_acts_shape} "
                    # f"host_act [{host_act_val}] "
                    # f"devs_and_acts [{devs_and_acts_val}] "
                    # f"host_act size: {host_act_size}, devs_and_acts size: {devs_and_acts_size}, total size: {total_size} "
                    + infer_msg)

        if num_total_updates % args.valid_freq == 0:
            valid_fn(num_total_updates)

    if num_total_updates % args.valid_freq != 0:
        valid_fn(num_total_updates)

    logging.info("Training done")


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    args = parse_args()
    Federation.init_federation(args.task_id, args.party, args.config_file)
    main(args)
    Federation.shutdown_federation()
