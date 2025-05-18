# -*- coding:utf-8 -*-
import math
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Dense, Dropout, Flatten
if tf.__version__.startswith('1.'):  # TensorFlow 1.x
    from tensorflow.keras.layers import BatchNormalization, LayerNormalization
    from tensorflow.keras.initializers import GlorotNormal
elif tf.__version__.startswith('2.'):
    from keras.layers import BatchNormalization, LayerNormalization
    GlorotNormal = tf.keras.initializers.GlorotNormal
    # from tensorflow.keras.initializers import GlorotNormal

from tensorflow.python.keras.regularizers import l2

from .sequence import AttentionSequencePoolingLayer, Transformer
from .neural_layers import MLP

class DIN:
    def __init__(self, guest_embedding_size=128, guest_num_buckets=5000,
                 host_embedding_size=128, host_num_buckets=5000,
                 dropout_rate=0.5, l2_reg=0.01,
                 guest_bottom_mlp_units=[256, 128, 96, 64],
                 host_bottom_mlp_units=[256, 128, 96, 64],
                 attention_pooling_units=[80, 40],
                 top_mlp_units=[200, 80],
                 **kwargs):
        self.guest_embedding_size = guest_embedding_size
        self.guest_num_buckets = guest_num_buckets
        self.host_embedding_size = host_embedding_size
        self.host_num_buckets = host_num_buckets
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Guest embedding and bottom model
        self.guest_embedding = Embedding(input_dim=guest_num_buckets, output_dim=guest_embedding_size,
                                         embeddings_initializer=GlorotNormal(), name="guest_embedding")
        self.guest_bottom_mlp = MLP(hidden_units=guest_bottom_mlp_units, activation='relu', dropout_rate=dropout_rate,
                                    l2_reg=l2_reg, name="guest_bottom_mlp")

        # Host embedding and bottom model
        self.host_embedding = Embedding(input_dim=host_num_buckets, output_dim=host_embedding_size,
                                        embeddings_initializer=GlorotNormal(), name="host_embedding")
        self.host_bottom_mlp = MLP(hidden_units=host_bottom_mlp_units, activation='relu', dropout_rate=dropout_rate,
                                   l2_reg=l2_reg, name="host_bottom_mlp")

        # Attention sequence pooling layer
        self.attention_pooling = AttentionSequencePoolingLayer(att_hidden_units=attention_pooling_units, att_activation='dice',
                                                               weight_normalization=True, supports_masking=True, name="attention_pooling")

        # Top model
        self.top_mlp = MLP(hidden_units=top_mlp_units, activation='dice', dropout_rate=dropout_rate,
                           l2_reg=l2_reg, name="top_mlp")
        self.top_output = Dense(1, activation='sigmoid', name="top_output")

        self.use_time_diff = False

    def feat_embedding(self, embedding_layer, feats):
        batch_size = tf.shape(feats)[0]
        num_feats = feats.shape[1]
        embedded_dim = embedding_layer.output_dim
        embedded = embedding_layer(feats)
        embedded = tf.reshape(embedded, (batch_size, num_feats * embedded_dim))
        return embedded

    def feat_mlp(self, mlp, embedded):
        embedded = BatchNormalization()(embedded)
        return mlp(embedded)

    def embedding(self, embedding_layer, mlp, sparse_feats, dense_feats):
        embedded = self.feat_embedding(embedding_layer, sparse_feats)
        if dense_feats is not None:
            dense_feats = tf.cast(dense_feats, dtype=tf.float32)
            embedded = tf.concat([embedded, dense_feats], axis=-1)
        return self.feat_mlp(mlp, embedded)

    def user_model(self, host_sparse, host_dense = None):
        return self.embedding(self.host_embedding, self.host_bottom_mlp, host_sparse, host_dense)

    def generate_history_mask(self, input_tensor):
        mask = tf.not_equal(input_tensor, -1)
        mask = tf.cast(mask, tf.bool)
        return mask

    def process_user_behaviors(self, user_behaviors_sparse, user_behaviors_dense=None):
        if len(user_behaviors_sparse.shape) != 3:
            raise ValueError("user_behaviors must have the shape of (batch_size, history_size, num_features)")

        batch_size, seq_len, num_sparse_feats = tf.shape(user_behaviors_sparse)[0], user_behaviors_sparse.get_shape()[1], user_behaviors_sparse.get_shape()[2]
        user_behaviors_sparse = tf.reshape(user_behaviors_sparse, (batch_size * seq_len, num_sparse_feats))
        if user_behaviors_dense is not None:
            batch_size, seq_len, num_dense_feats = tf.shape(user_behaviors_dense)[0], \
            user_behaviors_dense.get_shape()[1], user_behaviors_dense.get_shape()[2]
            user_behaviors_dense = tf.reshape(user_behaviors_dense, (batch_size * seq_len, num_dense_feats))
        else:
            user_behaviors_dense = None
        user_behaviors_embedded = self.embedding(self.guest_embedding, self.guest_bottom_mlp, user_behaviors_sparse, user_behaviors_dense)
        embedded_dim = user_behaviors_embedded.shape[-1]
        user_behaviors_embedded = tf.reshape(user_behaviors_embedded, (batch_size, seq_len, embedded_dim))

        return user_behaviors_embedded

    def item_model(self, labels, host_output, behavior_timestamps, guest_sparse, user_behaviors_sparse, user_behaviors_dense=None, guest_dense=None, time_diff=None):
        guest_output = self.embedding(self.guest_embedding, self.guest_bottom_mlp, guest_sparse, guest_dense)
        guest_output = tf.expand_dims(guest_output, axis=1)

        behavior_timestamps = self.generate_history_mask(behavior_timestamps)
        user_behaviors = self.process_user_behaviors(user_behaviors_sparse, user_behaviors_dense)
        host_output = tf.expand_dims(host_output, axis=1)

        weighted_behaviors = self.attention_pooling(
                                [guest_output, user_behaviors],
                                mask=[None, behavior_timestamps],
                                )

        combined_output = tf.concat([host_output, guest_output, weighted_behaviors], axis=-1)

        top_output = self.top_mlp(combined_output)

        top_output = tf.squeeze(top_output, axis=1)

        logits = self.top_output(top_output)

        # loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        #     labels, logits, from_logits=True, label_smoothing=0.1
        # ))
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
        acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(labels, logits))

        return {
            "logits": logits,
            "proba": tf.nn.sigmoid(logits),
            "loss": loss,
            "acc": acc,
        }

class BST(DIN):
    def __init__(self, guest_embedding_size=128, guest_num_buckets=5000,
                 host_embedding_size=128, host_num_buckets=5000,
                 dropout_rate=0.2, l2_reg=0.01,
                 num_transformer_layers=1, att_head_num=8,
                 guest_bottom_mlp_units=[256, 128, 96, 64],
                 host_bottom_mlp_units=[256, 128, 96, 64],
                 top_mlp_units=[128, 96, 64, 32, 16],
                 transformer_seed = None,
                 **kwargs):
        super(BST, self).__init__(guest_embedding_size, guest_num_buckets,
                                  host_embedding_size, host_num_buckets,
                                  dropout_rate, l2_reg,
                                  guest_bottom_mlp_units=guest_bottom_mlp_units,
                                  host_bottom_mlp_units=host_bottom_mlp_units,
                                  top_mlp_units=top_mlp_units,
                                  **kwargs)
        # Transformer for user behavior sequence
        self.transformer_seed = transformer_seed
        self.num_transformer_layers = num_transformer_layers
        self.att_head_num = att_head_num
        self.top_mlp = MLP(hidden_units=top_mlp_units, activation='relu', dropout_rate=dropout_rate,
                           l2_reg=l2_reg, use_bn=False, name="top_mlp")



    def item_model(self, labels, host_output, behavior_timestamps, guest_sparse, user_behaviors_sparse, user_behaviors_dense=None, guest_dense=None, time_diff=None):

        input_padding = tf.zeros((tf.shape(guest_sparse)[0], 1), dtype=tf.float32) # (batch_size, 1), used for padding
        if guest_dense is not None:
            guest_dense = tf.concat([guest_dense, input_padding], axis=1)
        if user_behaviors_dense is not None:
            time_diff_expanded = tf.expand_dims(time_diff, axis=-1)  # (batch_size, history_size, 1), used as values
            time_diff_expanded = tf.cast(time_diff_expanded, dtype=tf.float32)
            user_behaviors_dense = tf.concat([user_behaviors_dense, time_diff_expanded], axis=-1)

        guest_output = self.embedding(self.guest_embedding, self.guest_bottom_mlp, guest_sparse, guest_dense)
        guest_output = tf.expand_dims(guest_output, axis=1)


        user_behaviors = self.process_user_behaviors(user_behaviors_sparse, user_behaviors_dense)
        user_behaviors = tf.concat([guest_output, user_behaviors], axis=1) # (batch_size, history_size+1 , guest_embedding_size), used as keys

        behavior_padding = tf.ones((tf.shape(behavior_timestamps)[0], 1), dtype=tf.bool) # (batch_size, 1), used for padding
        behavior_masks = self.generate_history_mask(behavior_timestamps)
        behavior_masks = tf.concat([behavior_padding, behavior_masks], axis=1) # (batch_size, history_size + 1), used as keys_len


        transformer_output = user_behaviors
        for _ in range(self.num_transformer_layers):
            att_embedding_size = transformer_output.get_shape().as_list()[-1]//self.att_head_num
            transformer_layer = Transformer(att_embedding_size= att_embedding_size, head_num = self.att_head_num, dropout_rate=0.4,
                                            use_positional_encoding = False, use_res = True,
                                            use_feed_forward = True, use_layer_norm = True, blinding = True,
                                            supports_masking = True, output_type = None, seed = self.transformer_seed
                                            )
            transformer_output = transformer_layer([transformer_output, transformer_output], mask=[behavior_masks, behavior_masks])


        transformer_output = self.attention_pooling([guest_output, transformer_output], mask = [None, behavior_masks])

        # transformer_output = Flatten()(transformer_output)
        # transformer_output = tf.expand_dims(transformer_output, axis=1)

        host_output = tf.expand_dims(host_output, axis=1)

        # Concatenate features
        combined_output = tf.concat([host_output, guest_output, transformer_output], axis=-1)

        # Top MLP
        top_output = self.top_mlp(combined_output)

        top_output = tf.squeeze(top_output, axis=1)

        # Final output
        logits = self.top_output(top_output)

        # Loss and accuracy
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
        acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(labels, logits))

        return {
            "logits": logits,
            "proba": tf.nn.sigmoid(logits),
            "loss": loss,
            "acc": acc,
        }