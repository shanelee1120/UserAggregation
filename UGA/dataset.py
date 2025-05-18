# -*- coding:utf-8 -*-
import h5py
import numpy as np

from fdl.federation import Federation

import os, sys, traceback
import time
import logging
import multiprocessing

# _DATASET_DIR = os.environ["DATASET_DIR"]
_DATASET_DIR = os.path.join(os.getcwd(), "data")

def generate_sequence(user_seq, timestamp, window_size, padding_position="pre"):
    t_seq = [x for x in user_seq[0].split('') if x]
    v_seq = [x for x in user_seq[1].split('') if x]

    t_seq = np.array(t_seq, dtype=float)
    v_seq = np.array(v_seq, dtype=int)

    valid_indices = np.where(t_seq < timestamp)[0]

    if len(valid_indices) == 0:
        time_sequence = np.full(window_size, -1, dtype=type(timestamp))
        item_sequence = np.full(window_size, -2147483648, dtype=int)
        return item_sequence, time_sequence

    if len(valid_indices) >= window_size:
        time_sequence = t_seq[valid_indices[-window_size:]]
        item_sequence = v_seq[valid_indices[-window_size:]]
    else:
        pad_size = window_size - len(valid_indices)
        pad_time = np.full(pad_size, -1, dtype=type(timestamp))
        pad_item = np.full(pad_size, -2147483648, dtype=int)

        if padding_position == "pre":
            time_sequence = np.concatenate([pad_time, t_seq[valid_indices]])
            item_sequence = np.concatenate([pad_item, v_seq[valid_indices]])
        else:  # "post"
            time_sequence = np.concatenate([t_seq[valid_indices], pad_time])
            item_sequence = np.concatenate([v_seq[valid_indices], pad_item])

    return item_sequence, time_sequence


def generate_sequences_for_batch(item_user_ids, timestamps, seq, window_size):
    assert len(item_user_ids) == len(timestamps)
    batch_size = len(item_user_ids)
    item_sequences = np.full((batch_size, window_size), -2147483648, dtype=np.int32)
    time_sequences = np.full((batch_size, window_size), -1, dtype=type(timestamps[0]))

    for i in range(batch_size):
        user_id = item_user_ids[i]
        user_seq = seq[user_id]
        item_seq, time_seq = generate_sequence(user_seq, timestamps[i], window_size)
        item_sequences[i] = item_seq
        time_sequences[i] = time_seq

    return item_sequences, time_sequences

def embed_item_sequence(item_sequences, item_feats):
    batch_size, max_timesteps = item_sequences.shape
    num_items, num_item_feats = item_feats.shape
    batch_behaviors = np.zeros((batch_size, max_timesteps, num_item_feats), dtype=np.float32)

    for b_idx in range(batch_size):
        for t_idx in range(max_timesteps):
            item_id = item_sequences[b_idx, t_idx]
            if item_id != -2147483648:
                # we use -2147483648 as padding value
                if item_id < num_items:
                    batch_behaviors[b_idx, t_idx, :] = item_feats[item_id]
                else:
                    batch_behaviors[b_idx, t_idx, :] = np.zeros(num_item_feats, dtype=np.float32)
            else:
                batch_behaviors[b_idx, t_idx, :] = np.zeros(num_item_feats, dtype=np.float32)

    return batch_behaviors

class FederatedGuestDataset(object):
    def __init__(self, max_pack_size, batch_size, history_size,
                 train_user_ids, train_item_sparse_feats, train_item_dense_feats, train_u2i, train_seq,
                 valid_user_ids, valid_item_sparse_feats, valid_item_dense_feats, valid_u2i, valid_seq,
                 ):
        self.max_pack_size = max_pack_size
        self.batch_size = batch_size
        self.history_size = history_size
        assert batch_size % max_pack_size == 0, "Please set the batch size as a multiple of pack size"
        self.batch_user_size = batch_size // max_pack_size

        self.train_user_ids = train_user_ids
        self.valid_user_ids = valid_user_ids
        self.train_item_sparse_feats = train_item_sparse_feats
        self.valid_item_sparse_feats = valid_item_sparse_feats
        if train_item_dense_feats is not None or valid_item_dense_feats is not None:
            self.train_item_dense_feats = train_item_dense_feats
            self.valid_item_dense_feats = valid_item_dense_feats
        else:
            self.train_item_dense_feats = None
            self.valid_item_dense_feats = None
        self.train_u2i = train_u2i
        self.valid_u2i = valid_u2i
        self.train_seq = train_seq
        self.valid_seq = valid_seq
        train_user_cnts = np.array([len(t[0]) + len(t[1]) for t in self.train_u2i[self.train_user_ids]], dtype=np.int32)
        self.train_pos_ratio = np.sum([len(t[0]) for t in self.train_u2i[self.train_user_ids]]) / train_user_cnts.sum()
        valid_user_cnts = np.array([len(t[0]) + len(t[1]) for t in self.valid_u2i[self.valid_user_ids]], dtype=np.int32)
        self.user_prob = 1.0 * train_user_cnts / train_user_cnts.sum()

        if self.max_pack_size == 1:
            def flatten(user_ids, u2i, user_cnts):
                flattened_user_ids = np.repeat(user_ids, user_cnts)

                item_ids = []
                timestamps = []
                labels = []

                for sublist in u2i[user_ids]:
                    labels.extend(np.ones(len(sublist[0]), dtype=np.int32))
                    labels.extend(np.zeros(len(sublist[1]), dtype=np.int32))
                    for item in sublist:
                        for sub_item in item:
                            item_ids.append(sub_item[0])
                            timestamps.append(sub_item[1])

                flattened_item_ids = np.array(item_ids, dtype=np.int32)
                flattened_timestamps = np.array(timestamps, dtype=np.int32)
                flattened_labels = np.array(labels, dtype=np.int32)

                return flattened_user_ids, flattened_item_ids, flattened_labels, flattened_timestamps

            self.flattened_train_user_ids, self.flattened_train_item_ids, self.flattened_train_labels,self.flattened_timestamps=\
            flatten(self.train_user_ids, self.train_u2i, train_user_cnts)

    def train_iterator(self, max_updates, num_workers=10, buffer_queue_size=10000):
        Federation.send_async(self.train_pos_ratio, topic="train_pos_ratio")
        pool = multiprocessing.Pool(processes=num_workers)
        m = multiprocessing.Manager()
        queue = m.Queue(buffer_queue_size)
        stop_event = m.Queue(1)

        if self.max_pack_size > 1:
            data_workers = [pool.apply_async(
                FederatedGuestDataset._random_sampling,
                (self.batch_user_size, self.max_pack_size, max_updates, self.history_size,
                 self.train_user_ids, self.user_prob, self.train_u2i, self.train_seq,
                 queue, stop_event)) for _ in range(num_workers)]
        else:
            data_workers = [pool.apply_async(
                FederatedGuestDataset._flattened_random_sampling,
                (self.batch_size, max_updates, self.history_size,
                 self.flattened_train_user_ids,
                 self.flattened_train_item_ids,
                 self.flattened_train_labels,
                 self.flattened_timestamps,
                 self.train_seq,
                 queue, stop_event)) for _ in range(num_workers)]

        cnt = 0
        while cnt < max_updates:
            (batch_user_ids, batch_item_user_ids, batch_item_ids,
             batch_user_has_pos, batch_num_u2i,
             batch_behaviors, batch_behavior_timestamps, batch_timestamps,
             batch_labels) = queue.get()

            Federation.send_async(batch_user_ids, topic="train_batch_user_ids")
            Federation.send_async(batch_labels, topic="train_batch_labels")
            Federation.send_async(batch_num_u2i, topic="batch_num_u2i")

            batch_item_sparse_feats = self.train_item_sparse_feats[batch_item_ids]
            batch_item_dense_feats  = self.train_item_dense_feats[batch_item_ids] if self.train_item_dense_feats is not None else None

            cnt += 1

            batch_behaviors_sparse = embed_item_sequence(batch_behaviors, self.train_item_sparse_feats)
            batch_behaviors_dense = embed_item_sequence(batch_behaviors, self.train_item_dense_feats) if self.train_item_dense_feats is not None else None

            batch_labels = batch_labels.reshape(-1)

            yield (batch_item_sparse_feats, batch_item_dense_feats,
                   batch_user_has_pos, batch_num_u2i,
                   # batch_item_user_ids,
                   batch_item_ids,
                   batch_behaviors_sparse, batch_behaviors_dense,
                   batch_behavior_timestamps, batch_timestamps,
                   batch_labels,
                   self.train_pos_ratio)

    def valid_iterator(self):
        user_offset = 0
        while user_offset < self.valid_user_ids.shape[0]:
            start = user_offset
            end = min(start + self.batch_user_size, self.valid_user_ids.shape[0])
            batch_user_ids = self.valid_user_ids[start:end]
            user_offset = end

            batch_u2i = self.valid_u2i[batch_user_ids]
            batch_labels = []
            batch_num_u2i = []
            batch_item_ids = []
            batch_timestamps = []
            for v in batch_u2i:
                labels = np.concatenate([[1] * len(v[0]) + [0] * len(v[1])]).astype(np.int32)
                num_u2i = np.array([len(v[0]) + len(v[1])], dtype=np.int32)
                item_ids = np.concatenate([np.array([item[0] for item in v[0]] + [item[0] for item in v[1]])]).astype(np.int32)
                timestamps = np.concatenate([np.array([item[1] for item in v[0]] + [item[1] for item in v[1]])])
                batch_labels.append(labels)
                batch_num_u2i.append(num_u2i)
                batch_item_ids.append(item_ids)
                batch_timestamps.append(timestamps)
            batch_labels = np.concatenate(batch_labels).astype(np.int32)
            batch_num_u2i = np.concatenate(batch_num_u2i).astype(np.int32)
            batch_item_ids = np.concatenate(batch_item_ids).astype(np.int32)
            batch_timestamps = np.concatenate(batch_timestamps)
            
            batch_item_user_ids = np.concatenate(
                [np.full(batch_num_u2i[i], batch_user_ids[i]) for i in range(len(batch_user_ids))]
            )

            batch_behaviors, batch_behavior_timestamps = generate_sequences_for_batch(batch_item_user_ids, batch_timestamps, self.valid_seq, self.history_size)

            batch_behaviors_sparse = embed_item_sequence(batch_behaviors, self.valid_item_sparse_feats)
            batch_behaviors_dense = embed_item_sequence(batch_behaviors, self.valid_item_dense_feats) if self.valid_item_dense_feats is not None else None

            batch_item_sparse_feats = self.valid_item_sparse_feats[batch_item_ids]
            batch_item_dense_feats = self.valid_item_dense_feats[batch_item_ids] if self.valid_item_dense_feats is not None else None

            # batch_item_feats = np.concatenate([batch_item_feats, batch_item_ids.reshape(-1, 1)], axis=1) # TD1

            yield (batch_item_sparse_feats, batch_item_dense_feats,
                   batch_num_u2i,
                   # batch_item_user_ids,
                   batch_item_ids,
                   batch_behaviors_sparse, batch_behaviors_dense,
                   batch_timestamps, batch_behavior_timestamps,
                   batch_labels)

    @staticmethod
    def _random_sampling(batch_user_size, max_pack_size, max_updates, history_size,
                         user_ids, user_prob, u2i, seq, queue, stop_signal):
        proc_name = multiprocessing.current_process()
        while stop_signal.empty():
            try:
                batch_user_ids = np.random.choice(user_ids, batch_user_size, p=user_prob)
                batch_item_ids, batch_item_user_ids, batch_behaviors, batch_labels, batch_num_u2i, batch_user_has_pos, batch_behavior_timestamps, batch_timestamps = [], [], [], [], [], [], [], []

                # set_item_ids = set()
                for user_id in batch_user_ids:
                    user_pos, user_neg = u2i[user_id]
                    user_seq = seq[user_id]
                    num_pos = len(user_pos)
                    num_neg = len(user_neg)

                    indices = np.random.choice(
                        num_pos + num_neg,
                        size=min(max_pack_size, num_pos + num_neg),
                        replace=False,
                    )

                    user_item_ids = []
                    user_timestamps = []
                    item_user_ids = []
                    user_behaviors = []
                    user_behavior_timestamps = []

                    for idx in indices:
                        if idx < num_pos:
                            item_id, timestamp = user_pos[idx]
                        else:
                            item_id, timestamp = user_neg[idx - num_pos]
                        user_item_ids.append(item_id)

                        item_user_ids.append(user_id)
                        user_timestamps.append(timestamp)
                        item_seq, time_seq = generate_sequence(user_seq, timestamp, history_size)
                        user_behaviors.append(item_seq)
                        user_behavior_timestamps.append(time_seq)

                    user_labels = (indices < num_pos).astype(np.int32)
                    # batch_user_has_pos.append(np.any(user_labels).astype(np.int32))
                    batch_user_has_pos.append(np.sum(user_labels).astype(np.int32))
                    batch_item_ids.append(user_item_ids)
                    batch_behaviors.append(user_behaviors)
                    batch_behavior_timestamps.append(user_behavior_timestamps)
                    batch_labels.append(user_labels)
                    batch_timestamps.append(user_timestamps)
                    batch_num_u2i.append(indices.shape[0])
                    batch_item_user_ids.append(item_user_ids)

                batch_item_ids = np.concatenate(batch_item_ids).astype(np.int32)
                batch_behaviors = np.concatenate(batch_behaviors)
                batch_behavior_timestamps = np.concatenate(batch_behavior_timestamps)
                batch_labels = np.concatenate(batch_labels)
                batch_item_user_ids = np.concatenate(batch_item_user_ids)
                batch_user_has_pos = np.array(batch_user_has_pos, dtype=np.int32)
                batch_num_u2i = np.array(batch_num_u2i, dtype=np.int32)
                batch_timestamps = np.concatenate(batch_timestamps)

                queue.put((batch_user_ids, batch_item_user_ids, batch_item_ids,
                           batch_user_has_pos, batch_num_u2i,
                           batch_behaviors, batch_behavior_timestamps, batch_timestamps,
                           batch_labels))
            except Exception as e:
                msg = traceback.format_exc()
                print(f"Proc [{Proc}] met exception in random sampling: {msg}")
                break

    @staticmethod
    def _flattened_random_sampling(batch_size, max_updates, history_size,
                                   flattened_user_ids, flattened_item_ids, flattened_labels, flattened_timestamps, seq,
                                   queue, stop_signal):
        proc_name = multiprocessing.current_process()
        assert flattened_user_ids.shape[0] == flattened_item_ids.shape[0] == flattened_labels.shape[0]
        while stop_signal.empty():
            try:
                batch_indices = np.random.choice(flattened_user_ids.shape[0], batch_size)

                batch_user_ids = batch_item_user_ids = flattened_user_ids[batch_indices]
                batch_item_ids = flattened_item_ids[batch_indices]
                batch_labels   = flattened_labels[batch_indices]

                batch_timestamps = flattened_timestamps[batch_indices]
                batch_behaviors, batch_behavior_timestamps = \
                    (generate_sequences_for_batch(batch_user_ids, batch_timestamps, seq, history_size))
                batch_user_has_pos = batch_labels

                batch_num_u2i = np.ones(batch_size)

                queue.put((batch_user_ids, batch_item_user_ids, batch_item_ids,
                           batch_user_has_pos, batch_num_u2i,
                           batch_behaviors, batch_behavior_timestamps, batch_timestamps,
                           batch_labels))

            except Exception as e:
                msg = traceback.format_exc()
                print(f"Proc [{Proc}] met exception in random sampling: {msg}")
                break


class FederatedHostDataset(object):
    def __init__(self, max_pack_size, batch_size,
                 train_user_ids, train_user_sparse_feats, valid_user_dense_feats,
                 valid_user_ids, valid_user_sparse_feats, train_user_dense_feats):
        self.max_pack_size = max_pack_size
        self.batch_size = batch_size
        assert batch_size % max_pack_size == 0, "Please set the batch size as a multiple of pack size"
        self.batch_user_size = batch_size // max_pack_size

        self.train_user_ids = train_user_ids
        self.valid_user_ids = valid_user_ids
        self.train_user_sparse_feats = train_user_sparse_feats
        self.valid_user_sparse_feats = valid_user_sparse_feats

        if train_user_dense_feats is not None or valid_user_dense_feats is not None:
            self.train_user_dense_feats = train_user_dense_feats
            self.valid_user_dense_feats = valid_user_dense_feats
        else:
            self.train_user_dense_feats = None
            self.valid_user_dense_feats = None


    def train_iterator(self, max_updates):
        train_pos_ratio = Federation.next_object(topic="train_pos_ratio")
        cnt = 0
        while cnt < max_updates:
            # time_start = time.time()
            batch_user_ids = Federation.next_object(topic="train_batch_user_ids")
            batch_user_sparse_feats = self.train_user_sparse_feats[batch_user_ids]
            batch_user_dense_feats = self.train_user_dense_feats[batch_user_ids] if self.train_user_dense_feats is not None else None
            # batch_user_has_pos = Federation.next_object(topic="train_batch_user_has_pos")
            batch_labels = Federation.next_object(topic="train_batch_labels")
            batch_num_u2i = Federation.next_object(topic="batch_num_u2i")
            # logging.debug(f"Retrive batch data cost {time.time() - time_start:.6f}s")
            cnt += 1
            # yield batch_user_feats, batch_user_has_pos
            yield batch_user_sparse_feats, batch_user_dense_feats, batch_labels, batch_num_u2i, train_pos_ratio

    def valid_iterator(self):
        user_offset = 0
        while user_offset < self.valid_user_ids.shape[0]:
            start = user_offset
            end = min(start + self.batch_user_size, self.valid_user_ids.shape[0])
            batch_user_ids = self.valid_user_ids[start: end]
            user_offset = end
            batch_user_sparse_feats = self.valid_user_sparse_feats[batch_user_ids]
            batch_user_dense_feats = self.valid_user_dense_feats[batch_user_ids] if self.valid_user_dense_feats is not None else None
            yield batch_user_sparse_feats, batch_user_dense_feats


def load_data(args):
    dataset_dir = os.path.join(_DATASET_DIR, args.data)
    if args.data in ("movielens_1m","tenrec_article"):
        train_user_ids = valid_user_ids = np.load(os.path.join(dataset_dir, "user_ids.npy"), allow_pickle=True)
        if args.party == "guest":
            train_item_sparse_feats = valid_item_sparse_feats = np.load(os.path.join(dataset_dir, "item_feats.npy"), allow_pickle=True)
            train_item_dense_feats  = valid_item_dense_feats  = np.load(os.path.join(dataset_dir,  "item_dense_feats.npy"), allow_pickle=True) \
                if os.path.exists(os.path.join(dataset_dir, "item_dense_feats.npy")) else None
            train_u2i = np.load(os.path.join(dataset_dir, "train_u2i.npy"), allow_pickle=True)
            valid_u2i = np.load(os.path.join(dataset_dir, "valid_u2i.npy"), allow_pickle=True)
            train_seq = valid_seq = np.load(os.path.join(dataset_dir, "seq.npy"), allow_pickle=True)
        else:
            train_user_sparse_feats = valid_user_sparse_feats = np.load(os.path.join(dataset_dir, "user_feats.npy"), allow_pickle=True)
            train_user_dense_feats  = valid_user_dense_feats  = np.load(os.path.join(dataset_dir,  "user_dense_feats.npy"), allow_pickle=True) \
                if os.path.exists(os.path.join(dataset_dir, "user_dense_feats.npy")) else None
    elif args.data in ("ad_click", "kuairec"):
        if args.data == "kuairec":
            dataset_dir = os.path.join(_DATASET_DIR, "KuaiRec 2.0")
        train_user_ids = np.load(os.path.join(dataset_dir, "train_user_ids.npy"), allow_pickle=True)
        valid_user_ids = np.load(os.path.join(dataset_dir, "valid_user_ids.npy"), allow_pickle=True)
        if args.party == "guest":
            train_item_sparse_feats = valid_item_sparse_feats = np.load(os.path.join(dataset_dir, "item_sparse_feats.npy"), allow_pickle=True)
            train_item_dense_feats  = valid_item_dense_feats  = np.load(os.path.join(dataset_dir,  "item_dense_feats.npy"), allow_pickle=True) \
                if os.path.exists(os.path.join(dataset_dir, "item_dense_feats.npy")) else None
            train_u2i = np.load(os.path.join(dataset_dir, "train_u2i.npy"), allow_pickle=True)
            valid_u2i = np.load(os.path.join(dataset_dir, "valid_u2i.npy"), allow_pickle=True)
            train_seq = valid_seq = np.load(os.path.join(dataset_dir, "seq.npy"), allow_pickle=True)
        else:
            train_user_sparse_feats = valid_user_sparse_feats = np.load(os.path.join(dataset_dir, "user_sparse_feats.npy"),allow_pickle=True)
            train_user_dense_feats  = valid_user_dense_feats  = np.load(os.path.join(dataset_dir,  "user_dense_feats.npy"),allow_pickle=True) \
                if os.path.exists(os.path.join(dataset_dir, "user_dense_feats.npy")) else None
    else:
        raise ValueError(f"No such dataset: {args.data}")

    if args.party == "guest":
        return FederatedGuestDataset(
            args.pack_size, args.batch_size, args.history_size,
            train_user_ids, train_item_sparse_feats, train_item_dense_feats, train_u2i, train_seq,
            valid_user_ids, valid_item_sparse_feats, valid_item_dense_feats, valid_u2i, valid_seq)
    else:
        return FederatedHostDataset(
            args.pack_size, args.batch_size,
            train_user_ids, train_user_sparse_feats, valid_user_dense_feats,
            valid_user_ids, valid_user_sparse_feats, train_user_dense_feats)
