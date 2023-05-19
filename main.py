import os
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras import losses, models
from functools import lru_cache
import matplotlib.pyplot as plt
import cv2
import unet


# print(tf.config.list_physical_devices("GPU"))
# exit()


# @lru_cache
# def sinusemb(t, emb_dim):
#     k = tf.range(1, emb_dim / 2 + 1)
#     wk = tf.math.exp(-2. * k / emb_dim * tf.math.log(10000.)) * t
#
#     emb = tf.concat([tf.math.sin(wk), tf.math.cos(wk)], axis=-1)
#     emb = tf.repeat(emb, emb_dim)
#     emb = tf.reshape(emb, (emb_dim, emb_dim, 1))
#
#     return emb


batch_size = 8
im_s = 128
T = 16
logsize = 10
model = unet.unet(im_s)

# ps = tf.linspace(0, 1, T-1)
# u = 0
# s = 0.3
# ps = 1 / (s * tf.sqrt(2 * tf.pi)) * tf.exp(- (ps - u) ** 2 / (2 * s ** 2))
# ps = ps / tf.reduce_sum(ps)


# PATH = "/home/amankp/PycharmProjects/diff-transfer/"
# dir_path = [PATH + "ITS/clear/" + fname for fname in os.listdir(PATH + "ITS/clear/")]
#
#
# def prepare_img_():
#     img = pil.open(random.choice(dir_path))
#     img = tf.convert_to_tensor(img, dtype=tf.dtypes.float32)
#     img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
#     img = (img - 0.5) * 2
#     img = tfa.image.rotate(img, random.uniform(-0.1, 0.1))
#     img = tf.image.central_crop(img, 0.95)
#
#     img = tf.image.random_flip_left_right(img)
#     img = tf.image.resize(img, (im_s, im_s))
#
#     return img
#
#
# BUFFER = []
# def prepare_img():
#     while len(BUFFER) < batch_size:
#         pick = random.choice(dir_path)
#         img = pil.open(pick)
#         img = tf.convert_to_tensor(img, dtype=tf.dtypes.float32)
#         img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
#         img = (img - 0.5) * 2
#
#         for i in range(0, img.shape[0], im_s):
#             for j in range(0, img.shape[1], im_s):
#                 crop = img[i:i + im_s, j:j + im_s, :]
#                 if crop.shape == (im_s, im_s, 3):
#                     crop = tfa.image.rotate(crop, random.uniform(-0.1, 0.1))
#                     crop = tf.image.central_crop(crop, 0.95)
#
#                     crop = tf.image.random_flip_left_right(crop)
#                     crop = tf.image.resize(crop, (im_s, im_s))
#                     BUFFER.append(crop)
#
#     img = BUFFER.pop(random.randint(0, len(BUFFER) - 1))
#     return img
#
#
# betas = tf.linspace(1e-4, 0.02, T + 1)
# alphas = 1 - betas
# alpha_hats = tf.math.cumprod(alphas)
#
#
# def noisify(img, t):
#     noise = tf.random.normal(img.shape, 0., 1.)
#     noise = tf.clip_by_value(noise, -1., 1.)
#
#     xt = img * tf.sqrt(alpha_hats[t]) + noise * tf.sqrt(1 - alpha_hats[t])
#     xt = tf.clip_by_value(xt, -1., 1.)
#
#     return xt, noise


def normalize(arr):
    # return (arr - tf.reduce_min(arr)) / (tf.reduce_max(arr) - tf.reduce_min(arr))
    return arr / tf.reduce_max(arr)


Ts = tf.linspace(0., 1., T)
As = 0.3 * Ts + 0.7
Betas = 4. * Ts

PATH = "nyu-dv2/"
def get_pair(t=None):
    pick = PATH + random.choice(os.listdir(PATH)) + "/"
    pick = [pick + name.split(".")[0] for name in os.listdir(pick)]
    pick = list(set(pick))
    pick = random.choice(pick)
    img, depth = pick + ".jpg", pick + ".png"

    img, depth = cv2.imread(img), cv2.imread(depth)
    img, depth = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

    img, depth = tf.convert_to_tensor(img, dtype=tf.dtypes.float32), tf.convert_to_tensor(depth, dtype=tf.dtypes.float32)
    img, depth = normalize(img), normalize(depth)

    img, depth = tf.image.central_crop(img, 0.95), tf.image.central_crop(depth, 0.95)
    img, depth = tf.image.resize(img, (im_s, im_s)), tf.image.resize(depth, (im_s, im_s))

    t = t if t is not None else random.randint(1, T-1)

    # for t in range(T-1, -1, -1):
    #     if not os.path.exists("schedule"):
    #         os.mkdir("schedule")
    #     A, beta = As[t], betas[t]
    #
    #     t_x = tf.exp(-beta * depth)
    #     i_x = img * t_x + A * (1 - t_x)
    #     i_x = normalize(i_x)
    #
    #     plt.imshow(i_x)
    #     plt.savefig(f"schedule/{t}.png")
    # plt.close()
    # exit()

    A, beta = As[t], Betas[t]

    t_x = tf.exp(-beta * depth)
    i_x = img * t_x + A * (1 - t_x)
    i_x = normalize(i_x)

    img, i_x = (img - 0.5) * 2, (i_x - 0.5) * 2

    return img, i_x


def next_batch():
    batch = [get_pair() for _ in range(batch_size)]
    batch = tf.convert_to_tensor(batch)
    batch, xts = batch[:, 0], batch[:, 1]

    # noises = []
    # for img in batch:
    #     t = tf.random.choice(range(1, 3 * T // 4))
    #     temb = sinusemb(t, im_s)
    #
    #     for t in range(T, -1, -1):
    #         if not os.path.exists("schedule"):
    #             os.mkdir("schedule")
    #
    #         print(t)
    #         xt = fade(img, t)
    #         xt = tf.clip_by_value(xt, -1., 1.)
    #         plt.imshow(xt)
    #         plt.title(str(t))
    #         plt.savefig(f"schedule/{t}")
    #     exit()
    #
    #     xt = fade(img, t)
    #     xt = tf.concat([xt, temb], -1)
    #
    #     xts.append(xt)
    #     noises.append(noise)
    #
    # noises = tf.convert_to_tensor(noises)

    return batch, xts


def sample(i):
    t = random.randint(T // 2, T // 4 * 3)
    img, xt = get_pair(t)
    img, xt = tf.convert_to_tensor([img]), tf.convert_to_tensor([xt])

    # t = random.randint(T // 2, 3 * T // 4)
    # temb = sinusemb(t, im_s)
    # temb = tf.convert_to_tensor([temb])

    # xt = fade(img, t)
    # xt = tf.concat([xt, temb], -1)

    pred = model(xt)

    xt = tf.clip_by_value(xt, -1., 1.)
    pred = tf.clip_by_value(pred, -1., 1.)
    # noise = tf.clip_by_value(noise, -1., 1.)

    xt = (xt / 2) + 0.5
    pred = (pred / 2) + 0.5
    img = (img / 2) + 0.5
    # noise = (noise / 2) + 0.5

    _, ax = plt.subplots(1, 3)
    [ax[j].set_xticks([]) for j in range(3)] + [ax[j].set_yticks([]) for j in range(3)]
    ax[0].imshow(xt[0, :, :, :3], vmin=0, vmax=1)
    ax[1].imshow(img[0], vmin=0, vmax=1)
    ax[2].imshow(pred[0], vmin=0, vmax=1)

    if not os.path.exists("training_samples"):
        os.mkdir("training_samples")
    plt.savefig(f"training_samples/sample_{i}")
    plt.close()


# batch = next_batch(dir_path)
# _, ax = plt.subplots(1, 2)
# ax[0].imshow(batch[0][0, :, :, :3])
# ax[1].imshow(batch[1][0, :, :, :3])
# plt.show()


losses = []


def train(batches):
    lr = 1e-4
    opt = tf.optimizers.Adam(learning_rate=lr)
    loss = tf.losses.MeanSquaredError()

    for i in range(batches + 1):
        with tf.GradientTape() as gradtape:
            batch, xts = next_batch()

            pred = model(xts)
            batchloss = loss(batch, pred)
            losses.append(float(batchloss))

            grads = gradtape.gradient(batchloss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        if i % (logsize * 5) == 0:
            sample(i)

        if i % logsize * batch_size == 0 and i != 0:
            print(f"{i - logsize}->{i}: {tf.reduce_mean(losses[-logsize * batch_size:]):.6f}")

            if lr >= 1e-6:
                lr *= 0.99
                opt.learning_rate.assign(lr)

            if i % (logsize * 100) == 0:
                model.save(f"ckpts/model_ckpt{i}")


train(batches=10000000)
