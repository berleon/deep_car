from __future__ import division
import numpy as np
from six.moves import range
import PIL
from PIL import ImageEnhance


def data_generator(h5, batch_size=128, n_epoch=-1,
                   steering_history=[-8, -4, -2, -1],
                   delta_times=[0, 1, 2, 4, 8, 16],
                   shuffle=True):
    def get_steering(idx):
        idx = clip(idx)
        return np.array(steering[idx])

    def clip(x):
        return np.clip(x, 0, n-1)

    steering = np.array(h5['steering'])

    n = len(h5['image'])
    idx = np.arange(n)
    if n_epoch == -1:
        n_epoch = 1000000000

    for epoch in range(n_epoch):
        if shuffle:
            np.random.shuffle(idx)
        for b in range(0, n, batch_size):
            batch_idx = np.sort(idx[b:b+batch_size])
            s_m1 = get_steering(batch_idx - 1)
            batch = {
                'image': h5['image'][batch_idx, :, :],
                'speed': h5['speed'][batch_idx, :],
                'steering_abs': h5['steering'][batch_idx, :],
            }
            for t in delta_times:
                batch['steering_delta_{:02d}'.format(t)] = get_steering(batch_idx + t) - s_m1

            for t in steering_history:
                batch['steering_m{:02d}'.format(-t)] = get_steering(batch_idx + t)

            yield batch


def augment_img(img, crop_size=(64, 48)):
    width, height = img.size
    max_crop_left = img.size[0] - crop_size[0]
    max_crop_upper = img.size[1] - crop_size[1]
    left = np.random.choice(max_crop_left)
    upper = np.random.choice(max_crop_upper)
    img = img.crop([left, upper, left + crop_size[0], upper + crop_size[1]])
                    #width - left, height - upper])
    brightness = ImageEnhance.Brightness(img)

    img = brightness.enhance(np.random.uniform(0.75, 2.5))
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(np.random.uniform(0.75, 2.5))
    return img


def augment_batch(batch):
    images = [PIL.Image.fromarray(x) for x in batch["image"]]
    images = [augment_img(img) for img in images]
    batch_aug = {'image': np.stack([np.array(img) for img in images])}
    for k, v in batch.items():
        if k != 'image':
            batch_aug[k] = v
    return batch_aug


def labels_to_int(y, n_buckets=180):
    return (y * n_buckets).astype(np.int)


def discretize(y, n_buckets, min, max):
    y = np.clip(y, min, max)
    y_norm = (y - min) / (max - min)
    y_buckets = y_norm * n_buckets
    y_disc = y_buckets.astype(np.int)
    y_disc[y_disc == n_buckets] -= 1
    return y_disc


def get_steering_hist(batch):
    steering = []
    for key, arr in sorted(batch.items()):
        if 'steering_m' in key:
            steering.append(arr)
    return np.concatenate(steering, axis=-1)


def get_steering_delta(batch):
    steering = []
    for key, arr in sorted(batch.items()):
        if 'steering_delta' in key:
            steering.append(arr)
    return np.concatenate(steering, axis=-1)


def batch_to_numpy(batch, y_delta_buckets=9):
    x = 2. * batch['image'] / 255. - 1
    y_abs = batch['steering_abs']
    y_delta = get_steering_delta(batch)
    y_delta_disc = discretize(y_delta / np.pi * 180, n_buckets=y_delta_buckets, min=-45, max=45)
    return x[:, :, :, np.newaxis],  get_steering_hist(batch), y_delta_disc, y_abs


def continuous(y, n_buckets, min, max):
    y_norm = y / n_buckets
    y = y_norm * (max - min) + min + (max-min) / (2 * n_buckets)
    return y
