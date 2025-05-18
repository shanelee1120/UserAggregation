import numpy as np
from tensorflow.python.ops.init_ops import Zeros, Ones, TruncatedNormal, glorot_normal_initializer as glorot_uniform, \
    Constant
from keras.layers import Layer, Activation, Dropout, LayerNormalization
import tensorflow.compat.v1 as tf

from .neural_layers import MLP, ActivationUnit, ModifiedFeedForward
# from .normalization import LayerNormalization


class AttentionSequencePoolingLayer(Layer):
    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=True, **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            if not isinstance(input_shape, list) or len(input_shape) != 3:
                raise ValueError('A `attention_sequence_pooling_layer` layer should be called '
                                 'on a list of 3 inputs')

            if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
                raise ValueError(
                    "Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
                        len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))

            if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
                raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
                                 'inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
                                 'Got different shapes: %s' % (input_shape))
        else:
            pass
        self.local_att = ActivationUnit(
            self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024)
        super(AttentionSequencePoolingLayer, self).build(
            input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking==True, input must support masking")
            queries, keys = inputs
            key_masks = tf.expand_dims(mask[-1], axis=1)

        else:
            queries, keys, keys_length = inputs
            hist_len = keys.get_shape()[1]
            key_masks = tf.sequence_mask(keys_length, hist_len)

        attention_score = self.local_att([queries, keys], training=training)

        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        outputs._uses_learning_phase = training is not None

        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[1][1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):

        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PositionEncoding(Layer):
    def __init__(self, pos_embedding_trainable=True,
                 zero_pad=False,
                 scale=True, **kwargs):
        self.pos_embedding_trainable = pos_embedding_trainable
        self.zero_pad = zero_pad
        self.scale = scale
        super(PositionEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        _, T, num_units = input_shape.as_list()  # inputs.get_shape().as_list()
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * (i // 2) / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        if self.zero_pad:
            position_enc[0, :] = np.zeros(num_units)
        self.lookup_table = self.add_weight("lookup_table", (T, num_units),
                                            initializer=Constant(position_enc),
                                            trainable=self.pos_embedding_trainable)

        # Be sure to call this somewhere!
        super(PositionEncoding, self).build(input_shape)

    def call(self, inputs, mask=None):
        _, T, num_units = inputs.get_shape().as_list()
        position_ind = tf.expand_dims(tf.range(T), 0)
        outputs = tf.nn.embedding_lookup(self.lookup_table, position_ind)
        if self.scale:
            outputs = outputs * num_units ** 0.5
        return outputs + inputs

    def compute_output_shape(self, input_shape):

        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self, ):
        config = {'pos_embedding_trainable': self.pos_embedding_trainable, 'zero_pad': self.zero_pad,
                  'scale': self.scale}
        base_config = super(PositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# class Transformer(Layer):
#     def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=False,
#                  use_res=True,
#                  use_feed_forward=True, use_layer_norm=False, blinding=True, seed=1024, supports_masking=True,
#                  attention_type="scaled_dot_product", output_type="mean", **kwargs):
#         if head_num <= 0:
#             raise ValueError('head_num must be a int > 0')
#         self.att_embedding_size = att_embedding_size
#         self.head_num = head_num
#         self.num_units = att_embedding_size * head_num
#         self.use_res = use_res
#         self.use_feed_forward = use_feed_forward
#         self.seed = seed
#         self.use_positional_encoding = use_positional_encoding
#         self.dropout_rate = dropout_rate
#         self.use_layer_norm = use_layer_norm
#         self.blinding = blinding
#         self.attention_type = attention_type
#         self.output_type = output_type
#         super(Transformer, self).__init__(**kwargs)
#         self.supports_masking = supports_masking
#
#     def build(self, input_shape):
#         embedding_size = int(input_shape[0][-1])
#         if self.num_units != embedding_size:
#             raise ValueError(
#                 "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (
#                     self.att_embedding_size, self.head_num, embedding_size))
#         self.seq_len_max = int(input_shape[0][-2])
#         self.W_Query = self.add_weight(name='query',
#                                        shape=[embedding_size, self.att_embedding_size * self.head_num],
#                                        dtype=tf.float32,
#                                        initializer=TruncatedNormal(seed=self.seed))
#         self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
#                                      dtype=tf.float32,
#                                      initializer=TruncatedNormal(seed=self.seed + 1))
#         self.W_Value = self.add_weight(name='value',
#                                        shape=[embedding_size, self.att_embedding_size * self.head_num],
#                                        dtype=tf.float32,
#                                        initializer=TruncatedNormal(seed=self.seed + 2))
#         if self.attention_type == "additive":
#             self.b = self.add_weight('b', shape=[self.att_embedding_size], dtype=tf.float32,
#                                      initializer=glorot_uniform(seed=self.seed))
#             self.v = self.add_weight('v', shape=[self.att_embedding_size], dtype=tf.float32,
#                                      initializer=glorot_uniform(seed=self.seed))
#         elif self.attention_type == "ln":
#             self.att_ln_q = LayerNormalization()
#             self.att_ln_k = LayerNormalization()
#         # if self.use_res:
#         #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
#         #                                  initializer=TruncatedNormal(seed=self.seed))
#         if self.use_feed_forward:
#             self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
#                                        initializer=glorot_uniform(seed=self.seed))
#             self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
#                                        initializer=glorot_uniform(seed=self.seed))
#
#         self.dropout = Dropout(
#             self.dropout_rate, seed=self.seed)
#         self.ln = LayerNormalization()
#         if self.use_positional_encoding:
#             self.query_pe = PositionEncoding()
#             self.key_pe = PositionEncoding()
#         super(Transformer, self).build(input_shape)
#
#     def call(self, inputs, mask=None, training=None, **kwargs):
#
#         if self.supports_masking:
#             queries, keys = inputs
#             query_masks, key_masks = mask
#             query_masks = tf.cast(query_masks, tf.float32)
#             key_masks = tf.cast(key_masks, tf.float32)
#         else:
#             queries, keys, query_masks, key_masks = inputs
#
#             query_masks = tf.sequence_mask(
#                 query_masks, self.seq_len_max, dtype=tf.float32)
#             key_masks = tf.sequence_mask(
#                 key_masks, self.seq_len_max, dtype=tf.float32)
#             query_masks = tf.squeeze(query_masks, axis=1)
#             key_masks = tf.squeeze(key_masks, axis=1)
#
#         if self.use_positional_encoding:
#             queries = self.query_pe(queries)
#             keys = self.key_pe(keys)
#
#         Q = tf.tensordot(queries, self.W_Query,
#                          axes=(-1, 0))  # N T_q D*h
#         K = tf.tensordot(keys, self.W_key, axes=(-1, 0))
#         V = tf.tensordot(keys, self.W_Value, axes=(-1, 0))
#
#         # h*N T_q D
#         Q_ = tf.concat(tf.split(Q, self.head_num, axis=2), axis=0)
#         K_ = tf.concat(tf.split(K, self.head_num, axis=2), axis=0)
#         V_ = tf.concat(tf.split(V, self.head_num, axis=2), axis=0)
#
#         if self.attention_type == "scaled_dot_product":
#             # h*N T_q T_k
#             outputs = tf.matmul(Q_, K_, transpose_b=True)
#
#             outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
#         elif self.attention_type == "cos":
#             Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
#             K_cos = tf.nn.l2_normalize(K_, dim=-1)
#
#             outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # h*N T_q T_k
#
#             outputs = outputs * 20  # Scale
#         elif self.attention_type == 'ln':
#             Q_ = self.att_ln_q(Q_)
#             K_ = self.att_ln_k(K_)
#
#             outputs = tf.matmul(Q_, K_, transpose_b=True)  # h*N T_q T_k
#             # Scale
#             outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
#         elif self.attention_type == "additive":
#             Q_reshaped = tf.expand_dims(Q_, axis=-2)
#             K_reshaped = tf.expand_dims(K_, axis=-3)
#             outputs = tf.tanh(tf.nn.bias_add(Q_reshaped + K_reshaped, self.b))
#             outputs = tf.squeeze(tf.tensordot(outputs, tf.expand_dims(self.v, axis=-1), axes=[-1, 0]), axis=-1)
#         else:
#             raise ValueError("attention_type must be [scaled_dot_product,cos,ln,additive]")
#
#         key_masks = tf.tile(key_masks, [self.head_num, 1])
#
#         # (h*N, T_q, T_k)
#         key_masks = tf.tile(tf.expand_dims(key_masks, 1),
#                             [1, tf.shape(queries)[1], 1])
#
#         paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
#
#         # (h*N, T_q, T_k)
#
#         outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )
#         if self.blinding:
#             try:
#                 outputs = tf.matrix_set_diag(outputs, tf.ones_like(outputs)[
#                                                       :, :, 0] * (-2 ** 32 + 1))
#             except AttributeError:
#                 outputs = tf.compat.v1.matrix_set_diag(outputs, tf.ones_like(outputs)[
#                                                                 :, :, 0] * (-2 ** 32 + 1))
#
#         outputs -= tf.reduce_max(outputs, axis=-1, keep_dims=True)
#         outputs = tf.nn.softmax(outputs)
#         query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
#         # (h*N, T_q, T_k)
#         query_masks = tf.tile(tf.expand_dims(
#             query_masks, -1), [1, 1, tf.shape(keys)[1]])
#
#         outputs *= query_masks
#
#         outputs = self.dropout(outputs, training=training)
#         # Weighted sum
#         # ( h*N, T_q, C/h)
#         result = tf.matmul(outputs, V_)
#         result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)
#
#         if self.use_res:
#             # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
#             result += queries
#         if self.use_layer_norm:
#             result = self.ln(result)
#
#         if self.use_feed_forward:
#             fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
#             fw1 = self.dropout(fw1, training=training)
#             fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
#             if self.use_res:
#                 result += fw2
#             if self.use_layer_norm:
#                 result = self.ln(result)
#
#         if self.output_type == "mean":
#             return tf.reduce_mean(result, axis=1, keep_dims=True)
#         elif self.output_type == "sum":
#             return tf.reduce_sum(result, axis=1, keep_dims=True)
#         else:
#             return result
#
#     def compute_output_shape(self, input_shape):
#         return (None, 1, self.att_embedding_size * self.head_num)
#
#     def compute_mask(self, inputs, mask=None):
#         return None
#
#     def get_config(self, ):
#         config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num,
#                   'dropout_rate': self.dropout_rate, 'use_res': self.use_res,
#                   'use_positional_encoding': self.use_positional_encoding,
#                   'use_feed_forward': self.use_feed_forward,
#                   'use_layer_norm': self.use_layer_norm, 'seed': self.seed,
#                   'supports_masking': self.supports_masking,
#                   'blinding': self.blinding, 'attention_type': self.attention_type, 'output_type': self.output_type}
#         base_config = super(Transformer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

class Transformer(Layer):
    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=False,
                 use_res=True,
                 use_feed_forward=True, use_layer_norm=False, blinding=True, seed=1024, supports_masking=True,
                 attention_type="scaled_dot_product", output_type="latest", **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        super(Transformer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                    self.att_embedding_size, self.head_num, embedding_size))
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(name='query',
                                       shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value',
                                       shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.attention_type == "additive":
            self.b = self.add_weight('b', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=glorot_uniform(seed=self.seed))
            self.v = self.add_weight('v', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=glorot_uniform(seed=self.seed))
        elif self.attention_type == "ln":
            self.att_ln_q = LayerNormalization()
            self.att_ln_k = LayerNormalization()
        # if self.use_res:
        #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
        #                                  initializer=TruncatedNormal(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight(name='fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight(name='fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=glorot_uniform(seed=self.seed))
            self.fb1 = self.add_weight(name='fb1', shape=[4 * self.num_units], dtype=tf.float32,
                                       initializer=Zeros())
            self.fb2 = self.add_weight(name='fb2', shape=[self.num_units,], dtype=tf.float32,
                                       initializer=Zeros())
            # self.ffn = ModifiedFeedForward(num_units = self.num_units, dropout_rate=self.dropout_rate)

        self.dropout = Dropout(
            self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()
        if self.use_positional_encoding:
            self.query_pe = PositionEncoding()
            self.key_pe = PositionEncoding()
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs

            query_masks = tf.sequence_mask(
                query_masks, self.seq_len_max, dtype=tf.float32)
            key_masks = tf.sequence_mask(
                key_masks, self.seq_len_max, dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)

        if self.use_positional_encoding:
            queries = self.query_pe(queries)
            keys = self.key_pe(keys)

        Q = tf.tensordot(queries, self.W_Query,
                         axes=(-1, 0))  # N T_q D*h
        K = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        V = tf.tensordot(keys, self.W_Value, axes=(-1, 0))

        # h*N T_q D
        Q_ = tf.concat(tf.split(Q, self.head_num, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.head_num, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":
            # h*N T_q T_k
            outputs = tf.matmul(Q_, K_, transpose_b=True)

            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        elif self.attention_type == "cos":
            Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
            K_cos = tf.nn.l2_normalize(K_, dim=-1)

            outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # h*N T_q T_k

            outputs = outputs * 20  # Scale
        elif self.attention_type == 'ln':
            Q_ = self.att_ln_q(Q_)
            K_ = self.att_ln_k(K_)

            outputs = tf.matmul(Q_, K_, transpose_b=True)  # h*N T_q T_k
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        elif self.attention_type == "additive":
            Q_reshaped = tf.expand_dims(Q_, axis=-2)
            K_reshaped = tf.expand_dims(K_, axis=-3)
            outputs = tf.tanh(tf.nn.bias_add(Q_reshaped + K_reshaped, self.b))
            outputs = tf.squeeze(tf.tensordot(outputs, tf.expand_dims(self.v, axis=-1), axes=[-1, 0]), axis=-1)
        else:
            raise ValueError("attention_type must be [scaled_dot_product,cos,ln,additive]")

        key_masks = tf.tile(key_masks, [self.head_num, 1])

        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

        # (h*N, T_q, T_k)

        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )
        if self.blinding:
            outputs = tf.matrix_set_diag(outputs, tf.ones_like(outputs)[
                                                  :, :, 0] * (-2 ** 32 + 1))

        outputs -= tf.reduce_max(outputs, axis=-1, keep_dims=True)
        outputs = tf.nn.softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # (h*N, T_q, T_k)
        query_masks = tf.tile(tf.expand_dims(
            query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)

        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, V_)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
            result += queries
            result = self.dropout(result, training=training)
        if self.use_layer_norm:
            result = self.ln(result) # S' = LN(E+Dropout(MH(E)))

        if self.use_feed_forward:
            fw1 = tf.nn.leaky_relu(tf.tensordot(result, self.fw1, axes=[-1, 0])+ self.fb1) # S" = LeakyReLU(S'W1 + b1)
            fw2 = self.dropout(tf.tensordot(fw1, self.fw2, axes=[-1, 0])+ self.fb2, training=training) # S" = Dropout(LeakyReLU(S'W1 + b1)W2 + b2)
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

            # # here we exponentiate the result by the feed forward network defined in neural_layers.py
            # result = self.ffn(result)

            # the code here might be different from the BST architecture, so we won't use it
            # if self.use_feed_forward:
            #     fw1 = tf.nn.leaky_relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            #     fw1 = self.dropout(fw1, training=training)
            #     fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            #     if self.use_res:
            #         result += fw2
            #     if self.use_layer_norm:
            #         result = self.ln(result)

        if self.output_type == "mean":
            return tf.reduce_mean(result, axis=1, keep_dims=True)
        elif self.output_type == "sum":
            return tf.reduce_sum(result, axis=1, keep_dims=True)
        elif self.output_type == "latest":
            return result[:, 0, :]
        else:
            return result

    def compute_output_shape(self, input_shape):
        return (None, 1, self.att_embedding_size * self.head_num)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num,
                  'dropout_rate': self.dropout_rate, 'use_res': self.use_res,
                  'use_positional_encoding': self.use_positional_encoding,
                  'use_feed_forward': self.use_feed_forward,
                  'use_layer_norm': self.use_layer_norm, 'seed': self.seed,
                  'supports_masking': self.supports_masking,
                  'blinding': self.blinding, 'attention_type': self.attention_type, 'output_type': self.output_type}
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))