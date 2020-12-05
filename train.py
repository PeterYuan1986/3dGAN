import argparse
from tensorflow.python.data.experimental import prefetch_to_device
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras import *
from image_preprocess import *


class GAN():
    def __init__(self, width, height, depth, channel, dataset, dataset_name, batch_size, epochs, save_interval,
                 checkpoint_prefix, z=1000, ch_in=64, ):
        # Input shape
        self.dataset_name = dataset_name
        self.img_width = width
        self.img_height = height
        self.img_depth = depth
        self.img_channel = channel
        self.img_shape = (self.img_width, self.img_height, self.img_depth, self.img_channel)
        self.latent_dim = z
        self.ch_in = ch_in
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.checkpoint_prefix = checkpoint_prefix
        #optimizer =SGD(learning_rate=0.0001, momentum=0.0, nesterov=False)
        # Build and compile the discriminator
        optimizer = Adam(0.00002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
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
        self.combined.compile(loss='mse', optimizer=optimizer)

        """ Checkpoint """
        self.ckpt = tf.train.Checkpoint(discriminator=self.discriminator, generator=self.generator, optimizer=optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_prefix, max_to_keep=1)

        self.start_epoch = 0
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            self.start_epoch = int(self.manager.latest_checkpoint.split('-')[-1])
            print('Latest checkpoint restored!!')
        else:
            print('Not restoring from saved checkpoint')

        self.dataset = dataset

    def build_generator(self):
        ch_in = self.ch_in
        model = Sequential(name="Gnerator")
        model.add(Reshape((1, 1, 1, self.latent_dim), input_shape=(self.latent_dim,)))
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
        model.add(Conv3D(ch_in * 4, 3, strides=1, padding='same', name='conv3',
                         use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        model.add(Conv3D(ch_in*4, 3, strides=1, padding='same', name='conv4',
                         use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        model.add(Conv3D(ch_in *4, 3, strides=1, padding='same', name='conv5',
                         use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        # model.add(UpSampling3D(size=2, data_format=None))
        # model.add(Conv3D(ch_in*2, 3, strides=1, padding='same', name='conv6',
        #                  use_bias=False))
        # model.add(BatchNormalization(axis=-1))
        # model.add(ReLU())

        model.add(UpSampling3D(size=2, data_format=None))
        model.add(Conv3D(ch_in , 3, strides=1, padding='same', name='conv7',
                         use_bias=False))
        model.add(BatchNormalization(axis=-1))
        model.add(ReLU())

        model.add(UpSampling3D(size=2, data_format=None))
        model.add(Conv3D(1, 3, strides=1, padding='same', name='conv8',
                         use_bias=False))
        model.add(Activation("tanh"))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential(name="Discriminator")
        ch_in = self.ch_in
        model.add(Conv3D(ch_in, 4, strides=2, padding='same', name='conv1', input_shape=(64, 64, 64, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
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

        # model.add(Conv3D(ch_in * 8, 4, strides=2, padding='same', name='conv5'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))

        model.add(Conv3D(ch_in * 4, 3, strides=1, padding='same', name='conv6'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv3D(ch_in*2, 3, strides=1, padding='same', name='conv7'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv3D(1, 4, strides=1, padding='valid', name='conv8'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        img = Input(shape=(self.img_width, self.img_height, self.img_depth, self.img_channel,))
        validity = model(img)
        return Model(img, validity)

    def train(self):

        # Load the dataset
        X_train = self.dataset
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        for epoch in range(self.start_epoch, self.epochs):
            # ---------------------
            # Train Discriminator
            # ---------------------
            # Select a random half of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = next(X_train)
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
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
            noise1 = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise1, valid)

            # Plot the progress
            print("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
            epoch, self.epochs, d_loss[0] * 100, 100 * d_loss[1], g_loss * 100))
            if epoch % self.save_interval == 0 and epoch != 0:
                self.manager.save(checkpoint_number=epoch + 1)
                self.save_imgs(epoch)
        self.manager.save(checkpoint_number=self.epochs)

    def save_imgs(self, epoch):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        k=np.mean(noise)
        gen_img = self.generator.predict(noise)
        gen_imgs = postprocess_images(gen_img)
        tem=np.uint8(gen_imgs[0])
        new_image = nib.Nifti1Image(tem, affine=np.eye(4))
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

    parser.add_argument('--epochs', type=int, default='400000', help='epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--save_interval', type=int, default=2000, help='save_interval')

    parser.add_argument('--grw', type=int, default='10', help='gradient_penalty_weight: Lamda1')
    parser.add_argument('--lamda2', type=int, default='10', help='Lamda2 in G_loss')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate for all four model')
    parser.add_argument('--beta1', type=float, default=0.5, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--latentdimension', type=int, default=3000, help='latent dimension')
    parser.add_argument('--iteration', type=int, default=400000, help='total iteration')

    parser.add_argument('--g_iter', type=int, default=1, help='g_iter')
    parser.add_argument('--cd_iter', type=int, default=1, help='cd_iter')
    parser.add_argument('--d_iter', type=int, default=1, help='d_iter')
    parser.add_argument('--dataset', type=str, default='functional', help='dataset_name')
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
    checkpoint_prefix = os.path.join(checkpoint, "ckpt")

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
    data_set = dataset.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
    data_set = data_set.batch(args.batch_size, drop_remainder=True)
    data_set = data_set.apply(prefetch_to_device(gpu_device, buffer_size=AUTOTUNE))
    data_set_iter = iter(data_set)
    gan = GAN(args.img_width, args.img_height, args.img_depth, args.img_channel, data_set_iter,
              batch_size=args.batch_size, epochs=args.epochs, save_interval=args.save_interval,
              dataset_name=args.dataset, checkpoint_prefix=checkpoint_prefix, z=args.latentdimension)
    gan.train()


if __name__ == '__main__':
    main()
