# import tensorflow.compat.v1 as tf
from tensorflow.python.ops.init_ops import Zeros, Ones, glorot_normal_initializer
import tensorflow as tf
if tf.__version__.startswith('1.'):  # TensorFlow 1.x
    from tensorflow.keras.layers import Layer, Activation, Dropout, Dense, LayerNormalization
elif tf.__version__.startswith('2.'):
    from keras.layers import Layer, Activation, Dropout, Dense, LayerNormalization
    from keras.regularizers import l2
from keras import backend as K

# from .normalization import LayerNormalization

BatchNormalization = tf.keras.layers.BatchNormalization

from .activation import activation_layer

class MLP(Layer):
    """A simple multi-layer perceptron layer.
    """
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        super(MLP, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal_initializer(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(MLP, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PredictionLayer(Layer):
    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(1,), initializer=Zeros(), trainable=True)
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        if self.task == "binary":
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ActivationUnit(Layer):
    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(ActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `ActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError('A `ActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape[0], input_shape[1]))
        size = 4 * \
               int(input_shape[0][-1]
                   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=glorot_normal_initializer(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")
        self.mlp = MLP(self.hidden_units, self.activation, self.l2_reg, self.dropout_rate, self.use_bn, seed=self.seed)

        super(ActivationUnit, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        query, keys = inputs

        # keys_len = keys.get_shape()[1]
        # queries = K.repeat_elements(query, keys_len, 1)

        keys_len = tf.shape(keys)[1]
        queries = tf.broadcast_to(query, [tf.shape(query)[0], keys_len, tf.shape(query)[-1]])

        # print("query shape: ", query.shape)
        # print("queries shape: ", queries.shape)
        # print("keys shape: ", keys.shape)

        att_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = self.mlp(att_input, training=training)

        attention_score = tf.nn.bias_add(tf.tensordot(att_out, self.kernel, axes=(-1, 0)), self.bias)

        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(ActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ModifiedFeedForward(Layer):
    def __init__(self, num_units, dropout_rate=0.1, **kwargs):
        super(ModifiedFeedForward, self).__init__(**kwargs)
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def build(self, input_shape):
        self.w1 = self.add_weight(
            name='w1',
            shape=(input_shape[-1], 4 * self.num_units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b1 = self.add_weight(
            name='b1',
            shape=(4 * self.num_units,),
            initializer='zeros',
            trainable=True
        )

        self.w2 = self.add_weight(
            name='w2',
            shape=(4 * self.num_units, self.num_units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b2 = self.add_weight(
            name='b2',
            shape=(self.num_units,),
            initializer='zeros',
            trainable=True
        )
        super(ModifiedFeedForward, self).build(input_shape)

    def call(self, inputs, training=None):
        S = inputs  # S = result
        # S' = LayerNorm(S + Dropout(S))
        S_dropout = self.dropout(S, training=training)
        S_prime = self.layer_norm(S + S_dropout)

        # F = S' + Dropout(LeakyReLU(S'W1 + B1)W2 + B2)
        fc1_output = tf.matmul(S_prime, self.w1) + self.b1  # S'W1 + B1
        fc1_activated = tf.nn.leaky_relu(fc1_output)  # LeakyReLU(S'W1 + B1)
        fc1_dropout = self.dropout(fc1_activated, training=training)  # Dropout(LeakyReLU(S'W1 + B1))
        fc2_output = tf.matmul(fc1_dropout, self.w2) + self.b2  # (LeakyReLU(S'W1 + B1))W2 + B2
        fc2_dropout = self.dropout(fc2_output, training=training)  # Dropout(LeakyReLU(S'W1 + B1)W2 + B2)

        F = S_prime + fc2_dropout  # F = S' + Dropout(LeakyReLU(S'W1 + B1)W2 + B2)
        return F