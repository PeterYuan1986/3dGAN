import argparse
from tensorflow.python.data.experimental import prefetch_to_device
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras import *
from image_preprocess import *


class GAN():
    def __init__(self, width, height, depth, channel, dataset, ch_in=64, z=1000):
        # Input shape
        self.img_width = width
        self.img_height = height
        self.img_depth = depth
        self.img_channel = channel
        self.img_shape = (self.img_width, self.img_height, self.img_depth, self.img_channel)
        self.latent_dim = z
        self.ch_in = ch_in
        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # Build the generator
        self.generator = self.build_generator()
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated images as input and
        # determines validity
        valid = self.discriminator(img)
        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.dataset = np.array(dataset)

    def build_generator(self):
        ch_in = self.ch_in
        model = Sequential()
        model.add(Reshape((1, 1, 1, 1000), input_shape=(1000,)))
        model.add(Conv3DTranspose(ch_in * 8, kernel_size=4, strides=1, padding='valid',
                                                      use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        model.add(UpSampling3D(size=2, data_format=None))
        model.add(Conv3D(ch_in * 4, 3, strides=1, padding='same', name='conv2',
                                            use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        model.add(UpSampling3D(size=2, data_format=None))
        model.add(Conv3D(ch_in * 2, 3, strides=1, padding='same', name='conv3',
                                            use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        model.add(UpSampling3D(size=2, data_format=None))
        model.add(Conv3D(ch_in, 3, strides=1, padding='same', name='conv4',
                                            use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        model.add(UpSampling3D(size=2, data_format=None))
        model.add(Conv3D(1, 3, strides=1, padding='same', name='conv5',
                                            use_bias=False))
        model.add(Activation("tanh"))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        ch_in = self.ch_in
        model.add(Conv3D(ch_in, 4, strides=2, padding='same', name='conv1', input_shape=(64, 64, 64, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(ch_in * 2, 4, strides=2, padding='same', name='conv2'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv3D(ch_in * 4, 4, strides=2, padding='same', name='conv3'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv3D(ch_in * 8, 4, strides=2, padding='same', name='conv4'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv3D(ch_in * 8, 3, strides=1, padding='same', name='conv5'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        img = Input(shape=(self.img_width, self.img_height, self.img_depth, self.img_channel,))
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = self.dataset
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # ---------------------
            # Train Discriminator
            # ---------------------
            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            # Train the discriminator (real classified as ones
            # and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            # Train Generator
            # ---------------------
            # Train the generator (wants discriminator to mistake
            # images as real)
            g_loss = self.combined.train_on_batch(noise, valid)
        # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        new_image = nib.Nifti1Image(np.int16(gen_imgs[0]), affine=np.eye(4))
        name = self.dataset_name + '_epoch_' + str(epoch) + '.nii'
        sample_dir = './sample'
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        sample_dir = os.path.join(sample_dir, name)
        nib.save(new_image, sample_dir)


def parse_args():
    desc = "Tensorflow implementation of Alpha_WGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--img_width', type=int, default='64', help='img_width')
    parser.add_argument('--img_height', type=int, default='64', help='img_height')
    parser.add_argument('--img_depth', type=int, default='64', help='img_depth')
    parser.add_argument('--img_channel', type=int, default='1', help='img_channel')

    parser.add_argument('--epochs', type=int, default='1000', help='epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size')

    parser.add_argument('--grw', type=int, default='10', help='gradient_penalty_weight: Lamda1')
    parser.add_argument('--lamda2', type=int, default='10', help='Lamda2 in G_loss')
    parser.add_argument('--lr', type=int, default=1e-6, help='learning rate for all four model')
    parser.add_argument('--beta1', type=float, default=0.5, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--latentdimension', type=int, default=1000, help='latent dimension')
    parser.add_argument('--iteration', type=int, default=200000, help='total iteration')

    parser.add_argument('--g_iter', type=int, default=1, help='g_iter')
    parser.add_argument('--cd_iter', type=int, default=1, help='cd_iter')
    parser.add_argument('--d_iter', type=int, default=1, help='d_iter')
    parser.add_argument('--dataset', type=str, default='mri', help='dataset_name')
    parser.add_argument('--checkpoint_dir', type=str, default='model',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = './checkpoints'
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    checkpoint_dir = os.path.join(checkpoint, args.checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    dataset_name = args.dataset
    dataset_path = './dataset'
    if (args.phase == 'train'):
        datapath = os.path.join(dataset_path, dataset_name, 'train')
    else:
        datapath = os.path.join(dataset_path, dataset_name, 'test')

    img_class = Image_data(img_width=args.img_width, img_height=args.img_height, img_depth=args.img_depth,
                           dataset_path=datapath)
    img_class.preprocess()
    dataset = tf.data.Dataset.from_tensor_slices(img_class.dataset)
    dataset_num = len(img_class.dataset)  # all the images with different domain
    print("Dataset number : ", dataset_num)
    gpu_device = '/gpu:0'
    dataset = dataset.apply(prefetch_to_device(gpu_device, buffer_size=AUTOTUNE))
    gan = GAN(args.img_width, args.img_height, args.img_depth, args.img_channel, img_class.dataset, z=args.latentdimension)
    gan.train(epochs=args.epochs, batch_size=args.batch_size, save_interval=50)


if __name__ == '__main__':
    main()
