import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_addons.layers as layers_a
from keras import layers, Model, Sequential
from keras.utils import plot_model


C = 8


# class ReflectPad(layers.Layer):
#     def __init__(self, padding):
#         self.padding = padding
#         super(ReflectPad, self).__init__()
#
#     def __call__(self, x):
#         return tf.pad(x, [[0, 0], [self.padding[0]] * 2, [self.padding[1]] * 2, [0, 0]], 'REFLECT')


class ConvBlock(layers.Layer):
    def __init__(self, f):
        super(ConvBlock, self).__init__()

        self.stack = Sequential([
            # ReflectPad((1, 1)),
            layers.Conv2D(f * C, 5, 1, activation="tanh", padding="same"),
            layers.Normalization(),
            # ReflectPad((1, 1)),
            layers.Conv2D(f * C, 3, 1, activation="tanh", padding="same"),
            layers.Normalization(),
        ])

    def __call__(self, x):
        x0 = self.stack(x)
        x = layers.Concatenate()([x, x0])

        return x


class Downsample(layers.Layer):
    def __init__(self, f, use_attn=False):
        super(Downsample, self).__init__()

        self.use_attn = use_attn

        self.stack = Sequential([
            ConvBlock(f),
            ConvBlock(f),
            # ReflectPad((1, 1)),
            layers.Conv2D(f * C, 4, 2, activation="tanh", padding="same"),
            layers.Normalization(),
            ConvBlock(f),
            ConvBlock(f),
        ])

        # if use_attn:
        #     self.attn = SelfAttention()

    def __call__(self, x):
        # if self.use_attn:
        #     x = self.attn(x)

        x = self.stack(x)

        return x


class Upsample(layers.Layer):
    def __init__(self, f, use_attn=False):
        super(Upsample, self).__init__()

        self.use_attn = use_attn

        self.stack = Sequential([
            ConvBlock(f),
            ConvBlock(f),
            layers.Conv2DTranspose(f * C, 4, 2, padding="same", activation="tanh"),
            ConvBlock(f),
            ConvBlock(f),
        ])

        # if use_attn:
        #     self.attn = SelfAttention()

    def __call__(self, x):
        # if self.use_attn:
        #     x = self.attn(x)

        x = self.stack(x)

        return x


def unet(im_s):
    inputs = layers.Input((im_s, im_s, 3))

    # x = ReflectPad((1, 1))(inputs)
    x = ConvBlock(8)(inputs)

    skips = [x]

    filter_facs = [3, 3, 3, 4, 4, 4, 5]
    attns = [False, False, False, False, False, False, False, False, False]

    downs = [Downsample(f, attn) for f, attn in zip(filter_facs, attns)]
    ups = [Upsample(f, attn) for f, attn in zip(reversed(filter_facs), reversed(attns))]

    for down in downs:
        x = down(x)
        skips.append(x)

    for up, skip in zip(ups, reversed(skips[:-1])):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = layers.Concatenate()([x, inputs])
    x = ConvBlock(8)(x)

    x = layers.Conv2D(3, 1, 1, activation="tanh", padding="same")(x)
    outputs = layers.Normalization()(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    plot_model(model, "unet.png", show_shapes=True)

    return model

