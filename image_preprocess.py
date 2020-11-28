import numpy as np
import os
import nibabel as nib
import tensorflow as tf
from glob import glob
import skimage.transform as skTrans


class Image_data:
    def __init__(self, img_width, img_height, img_depth, dataset_path):
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.dataset_path = dataset_path
        self.dataset = []

    def image_to_tf(self, img_path):
        nib_img = nib.load(img_path)
        img_ary = nib_img.get_fdata()
        img = resize(img_ary, self.img_width, self.img_height, self.img_depth)
        y = np.expand_dims(img, axis=3)
        x_decode = tf.convert_to_tensor(y, dtype='float32')
        return x_decode

    def preprocess(self):
        image_list = glob(os.path.join(self.dataset_path) + '/*.nii.gz') + glob(
            os.path.join(self.dataset_path) + '/*.nii')
        self.dataset.extend(image_list)

        for idx, path in enumerate(self.dataset):
            image = self.image_to_tf(path)
            image = preprocess_fit_train_image(image)
            self.dataset[idx] = image

def resize(img, dx, dy, dz):
    return skTrans.resize(img, (dx, dy, dz), order=1, preserve_range=True)


def adjust_dynamic_range(images, range_in=(0, 255), range_out=(-1, 1),
                         out_dtype='float32'):  # preprocess, change 255 into -1 to 1.
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images


def preprocess_fit_train_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    return images


def postprocess_images(images):
    images = tf.squeeze(images,[-1]) # [1, 2, 3, 1]
    images = adjust_dynamic_range(images, range_in=(-1, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.cast(images, dtype=tf.dtypes.int16)
    return images
