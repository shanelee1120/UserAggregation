import tensorflow.compat.v1 as tf
tf.keras.backend.set_floatx('float64')


def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
          alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float64)
          input_shape = list(_x.get_shape())

          reduction_axes = list(range(len(input_shape)))
          del reduction_axes[axis]
          broadcast_shape = [1] * len(input_shape)
          broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x

def parametric_relu(_x, name):
    alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                              initializer=tf.constant_initializer(0.0),
                              dtype=tf.float64)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def activation_unit(guest_output_repeated, user_behaviors, history_mask, device="/gpu:0"):
    with tf.variable_scope("activation_unit", dtype=tf.float64, reuse=tf.AUTO_REUSE,
                           initializer=tf.glorot_normal_initializer()):
        with tf.device(device):
            batch_size = tf.shape(user_behaviors)[0]
            history_size = user_behaviors.get_shape()[1]

            # guest_output_expanded = tf.expand_dims(guest_output, axis=1)
            # guest_output_repeated = tf.tile(guest_output_expanded, [1, history_size, 1])

            interaction = tf.multiply(guest_output_repeated, user_behaviors)

            concat_input = tf.concat([guest_output_repeated, user_behaviors, interaction], axis=-1)

            hidden = dice(concat_input, name="dice_activation")

            attention_weights = tf.keras.layers.Dense(1, activation=None, name='attention_weights')(hidden)
            attention_weights = tf.reshape(attention_weights, [batch_size, history_size])

            attention_weights = tf.where(tf.not_equal(history_mask, -1),attention_weights,tf.fill(tf.shape(attention_weights), tf.float64.min))
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)

            weighted_behaviors = tf.multiply(
                attention_weights[:, :, None],
                user_behaviors)
            user_interests = tf.reduce_sum(weighted_behaviors, axis=1)

            return user_interests