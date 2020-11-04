# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import tensorlayer as tl
import keras_contrib

Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
BatchNorm = tf.keras.layers.BatchNormalization
Con2D_trans = tf.keras.layers.Conv2DTranspose
instance_norm = tl.layers.InstanceNorm2d

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

################################################################################################################################


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================
class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class Pad(tf.keras.layers.Layer):

    def __init__(self, paddings, mode='CONSTANT', constant_values=0, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, mode=self.mode, constant_values=self.constant_values)

def instance_norm(input):           # ?Ҿ?????(tensorflow 2.0 ???? ?????? instance normalization?? ???????? ?? ?????ؾ? ?? ?Ͱ???.)
    depth = input.get_shape()[3]

    #scale = tf.Variable([depth], tf.random_normal_initializer(1.0, 0.02))       # dtype is string(need to convert to float32)
    #scale = tf.cast(scale, tf.float32)
    #offset = tf.Variable([depth], tf.constant_initializer(0.0))                 # dtype is string(need to convert to float32)
    #offset = tf.cast(offset, tf.float32)

    scale = tf.compat.v1.get_variable("scale", [depth], initializer=tf.compat.v1.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = tf.compat.v1.get_variable("offset", [depth], initializer=tf.compat.v1.constant_initializer(0.0))

    mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (input - mean) * inv
    
    return (scale * normalized + offset)       # ?? ?κ??? ?Ҿ????? ?Ͱ???.

def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)
    
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        p = int((3 - 1) / 2)

        h = Pad([[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

        h = Pad([[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')(h)
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)

        return tf.keras.layers.add([x, h])

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)
    
    # 1
    h = Pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
    h = tf.keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.ReLU()(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)
    fla = tf.keras.layers.Flatten()(h)
    logits = tf.keras.layers.Dense(54)(fla)

    # 4
    #for _ in range(n_downsamplings):
    #    dim //= 2
    #    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
    #    h = InstanceNormalization(epsilon=1e-5)(h)
    #    h = tf.keras.layers.ReLU()(h)
    dim //= 2
    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)
    dim //= 2
    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)


    # 5
    h = Pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
    h = tf.keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.keras.layers.Activation('tanh')(h)

    return tf.keras.Model(inputs=inputs, outputs=[h, logits])


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    fla = tf.keras.layers.Flatten()(h)
    logits = tf.keras.layers.Dense(54)(fla)

    return tf.keras.Model(inputs=inputs, outputs=[h, logits])


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

class ItemPool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)

################################################################################################################################

################################################################################################################################

# ===========================================================
# =                     tensorlayer                         =
# ===========================================================

# ?ϴ??? tensorlayer?? ?̿??Ͽ? Model?? ?????غ???!!!!!
# tensorflow 2.0 layer?? tensorlayer 2.0?? layer?? ???? ȣȯ?? ?? ???°Ͱ???.(?????? ???? ?????غ?????)

#def ResnetGenerator(input_shape=(256, 256, 3),
#                    output_channels=3,
#                    dim=64,
#                    n_downsamplings=2,
#                    n_blocks=9,
#                    norm='instance_norm'):
#    Norm = tl.layers.InstanceNorm2d()
    
#    def _residual_block(x):
#        dim = x.shape[-1]
#        h = x

#        p = int((3 - 1) / 2)

#        h = tl.layers.PadLayer([[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')(h)
#        h = tl.layers.Conv2d(dim, (3,3), strides=(1,1), act=tf.nn.relu, padding='VALID', b_init=None)(h)
#        h = tl.layers.InstanceNorm2d()(h)

#        h = tl.layers.PadLayer([[0, 0], [p, p], [p, p], [0, 0]], mode='REFLECT')(h)
#        h = tl.layers.Conv2d(dim, (3,3), strides=(1,1), act=tf.nn.relu, padding='VALID', b_init=None)(h)
#        h = tl.layers.InstanceNorm2d()(h)
        
#        out = x + h
#        return out

#    # 0
#    h = inputs = tl.layers.Input(shape=input_shape)
    
#    # 1
#    h = tl.layers.PadLayer([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
#    h = tl.layers.Conv2d(dim, (7,7), strides=(1,1), act=tf.nn.relu, padding='VALID', b_init=None)(h)
#    h = tl.layers.InstanceNorm2d()(h)
    
#    # 2
#    for _ in range(n_downsamplings):
#        dim *= 2
#        h = tl.layers.Conv2d(dim, (3,3), strides=(2,2), act=tf.nn.relu, padding='SAME', b_init=None)(h)
#        h = tl.layers.InstanceNorm2d()(h)

#    # 3
#    for _ in range(n_blocks):
#        h = _residual_block(h)
    
#    # 4
#    for _ in range(n_downsamplings):
#        dim //= 2
#        h = tl.layers.DeConv2dLayer(dim, (3,3), strides=(2,2), act=tf.nn.relu, padding='SAME', b_init=None)(h)   # ???⼭???? ?ٽ? ?????ؾ??Ѵ?.
#        h = tl.layers.InstanceNorm2d()(h)

#    # 5
#    h = tl.layers.PadLayer([[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')(h)
#    h = tl.layers.Conv2d(output_channels, (7,7), strides=(1,1), act=tf.nn.tanh, padding='VALID')(h)

#    return tf.keras.Model(inputs=inputs, outputs=h)


#def ConvDiscriminator(input_shape=(256, 256, 3),
#                      dim=64,
#                      n_downsamplings=2,
#                      norm='instance_norm'):
#    dim_ = dim
#    Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

#    # 0
#    h = inputs = tf.keras.Input(shape=input_shape)

#    # 1
#    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
#    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

#    for _ in range(n_downsamplings - 1):
#        dim = min(dim * 2, dim_ * 8)
#        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
#        h = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)(h)
#        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

#    # 2
#    dim = min(dim * 2, dim_ * 8)
#    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
#    h = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)(h)
#    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

#    # 3
#    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

#    return tf.keras.Model(inputs=inputs, outputs=h)


## ==============================================================================
## =                          learning rate scheduler                           =
## ==============================================================================

#class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
#    # if `step` < `step_decay`: use fixed learning rate
#    # else: linearly decay the learning rate to zero

#    def __init__(self, initial_learning_rate, total_steps, step_decay):
#        super(LinearDecay, self).__init__()
#        self._initial_learning_rate = initial_learning_rate
#        self._steps = total_steps
#        self._step_decay = step_decay
#        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

#    def __call__(self, step):
#        self.current_learning_rate.assign(tf.cond(
#            step >= self._step_decay,
#            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
#            false_fn=lambda: self._initial_learning_rate
#        ))
#        return self.current_learning_rate

#class ItemPool:
#    def __init__(self, pool_size=50):
#        self.pool_size = pool_size
#        self.items = []

#    def __call__(self, in_items):

#        if self.pool_size == 0:
#            return in_items

#        out_items = []
#        for in_item in in_items:
#            if len(self.items) < self.pool_size:
#                self.items.append(in_item)
#                out_items.append(in_item)
#            else:
#                if np.random.rand() > 0.5:
#                    idx = np.random.randint(0, len(self.items))
#                    out_item, self.items[idx] = self.items[idx], in_item
#                    out_items.append(out_item)
#                else:
#                    out_items.append(in_item)
#        return tf.stack(out_items, axis=0)

################################################################################################################################