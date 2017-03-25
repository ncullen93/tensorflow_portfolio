"""
GAN in Keras where you only train on all fake or all real images
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc

import keras
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import (Input, Dense, Convolution2D,
    Deconvolution2D, BatchNormalization, Activation, Flatten,
    LeakyReLU, Reshape)


def load_mnist(image_dim_ordering='tf'):
    def normalization(X):
        """normalize between -1 and 1"""
        return X / 127.5 - 1
    from keras.datasets.mnist import load_data
    from keras.utils import np_utils
    (X_train, y_train), (X_test, y_test) = load_data()

    if image_dim_ordering == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train)
    X_test = normalization(X_test)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #X_train = np.lib.pad(X_train, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1))
    #X_test  = np.lib.pad(X_test, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1))
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return X_train, Y_train, X_test, Y_test

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.


def generator_model(g_input):
    """
    Takes a random vector z as input, and outputs a 2D image
    """
    g_fc1       = Dense(1024, activation='relu')(g_input)
    g_fc1_bn    = BatchNormalization()(g_fc1)

    g_fc2       = Dense(7*7*128,activation='relu')(g_fc1_bn)
    g_fc2_bn    = BatchNormalization()(g_fc2)

    # 7x7
    g_reshape   = Reshape((7,7,128))(g_fc2_bn)

    # 14x14
    g_conv1     = Deconvolution2D(64, 4, 4, subsample=(2,2), 
        output_shape=(None, 14,14,64), border_mode='same')(g_reshape)
    g_conv1     = BatchNormalization()(g_conv1)
    g_conv1     = Activation('relu')(g_conv1)

    # 28x28
    g_out       = Deconvolution2D(1, 5, 5, subsample=(1,1), 
        output_shape=(None, 28,28,1), border_mode='same')(g_conv1)
    g_tanh      = Activation('tanh')(g_out)

    return g_tanh


def discriminator_model(g_output, r_input):
    """
    Takes a 2D image as input, outputs a 2-value softmax classification representing
    the discriminators decision of the input image being a generated or real

    """
    ### CREATE SHARED LAYERS ###
    # first conv block
    shared_layers = []
    shared_layers.append(Convolution2D(64, 4, 4, subsample=(2,2), border_mode='same', name='disc_conv1'))
    shared_layers.append(BatchNormalization(name='disc_conv1_bn'))
    shared_layers.append(LeakyReLU(0.1, name='disc_conv1_act'))
    # second conv block
    shared_layers.append(Convolution2D(128, 4, 4, subsample=(2,2), border_mode='same', name='disc_conv2'))
    shared_layers.append(BatchNormalization(name='disc_conv2_bn'))
    shared_layers.append(LeakyReLU(0.1, name='disc_conv2_act'))
    # dense layers
    shared_layers.append(Flatten(name='disc_flat'))
    shared_layers.append(Dense(1024, name='disc_dense1'))
    shared_layers.append(BatchNormalization(name='disc_dense1_bn'))
    shared_layers.append(LeakyReLU(0.2, name='disc_dense1_act'))
    shared_layers.append(Dense(1, activation='sigmoid', name='disc_out'))

    ### REAL IMAGE NETWORK
    r_out = shared_layers[0](r_input)
    for layer in shared_layers[1:]:
        r_out = layer(r_out)

    ### FAKE IMAGE NETWORK
    f_out = shared_layers[0](g_output)
    for layer in shared_layers[1:]:
        f_out = layer(f_out)

    return r_out, f_out

## MODEL PARAMS 
z_dim       = 100
img_shape   = (28,28,1)

NB_EPOCH    = 100
NB_SAMPLES  = 60000
BATCH_SIZE  = 64

def generator_sample(z_dim, batch_size):
    return np.random.uniform(-1,1, (batch_size,z_dim))


## CREATE MODELS 
gen_input   = Input(shape=(z_dim,), name='generator_input')
fake_img    = generator_model(gen_input)
real_img    = Input(shape=(28,28,1), name='real_disc_input')

r_out,f_out = discriminator_model(fake_img, real_img)

fake_img_generator = Model(input=gen_input, output=fake_img)
d_model = Model(input=real_img, output=r_out)
g_model = Model(input=gen_input, output=f_out)

## COMPILE MODELS
d_opt   = Adam(lr=2e-4)
d_model.compile(loss='binary_crossentropy', optimizer=d_opt)

g_opt   = Adam(lr=1e-3)
g_model.compile(loss='binary_crossentropy', optimizer=g_opt)

def freeze_model(model):
    for l in model.layers:
        l.trainable=False

def unfreeze_model(model):
    for l in model.layers:
        l.trainable=True

## TRAIN MODELS 
X_train, y_train, X_test, y_test = load_mnist()
X_test = X_test[:500]
print('Training Model')
for epoch in range(NB_EPOCH):
    print(epoch)

    for batch_idx in range(int(np.ceil(NB_SAMPLES/BATCH_SIZE))):

        ## create discriminator inputs (alternate btwn real images & fake images)
        if batch_idx % 2 == 0:
            x_disc  = X_train[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            y_disc  = np.ones(x_disc.shape[0])
        else:
            x_disc  = fake_img_generator.predict(generator_sample(z_dim, BATCH_SIZE))
            y_disc  = np.zeros(x_disc.shape[0])
        
        # train discriminator to distinguish between real and fake images
        d_model.train_on_batch( x_disc, y_disc)
        ## freeze discriminator
        freeze_model(d_model)

        ## train generator to fool the discriminator
        z_sample = generator_sample(z_dim, BATCH_SIZE)
        g_model.train_on_batch( z_sample , np.ones(z_sample.shape[0]) )
        
        ## unfreeze discriminator
        unfreeze_model(d_model)

        if batch_idx % 20 == 0:
            # generate some fake images
            z_sample    = generator_sample(z_dim, BATCH_SIZE)
            fake_imgs   = fake_img_generator.predict(z_sample)
            fake_imgs   = (fake_imgs[:,:,:,0] + 1.) / 2.
            
            # save fake images to file
            for i in range(fake_imgs.shape[0]):
                scipy.misc.imsave('figures/prediction-%i-%i.jpg'%(epoch,i), fake_imgs[i])

            # evaluate losses on some data
            g_loss = g_model.evaluate( generator_sample(z_dim, BATCH_SIZE), np.ones(BATCH_SIZE), verbose=0)
            d_loss_real = d_model.evaluate( X_train[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE], 
                np.ones(BATCH_SIZE), verbose=0)
            d_loss_fake = d_model.evaluate( fake_img_generator.predict(generator_sample(z_dim, BATCH_SIZE)),
                np.zeros(x_disc.shape[0]), verbose=0)
            print('G LOSS: %.04f , D LOSS: %.04f' %(g_loss, (d_loss_real+d_loss_fake)))






