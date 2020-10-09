#Copyright (C) 2019  Soeren Lukassen

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This module contains the main code for generating and fitting a class-aware variational autoencoder.

"""

import json
import os
import sys
import argparse

import resVAE.utils as cutils

import numpy as np
import pandas as pd
from keras import backend as K
from keras import callbacks
from keras import layers
from keras import losses
from keras import regularizers
from keras.models import Model, model_from_json, load_model
from keras.utils import plot_model, multi_gpu_model, normalize
from keras.applications.vgg19 import VGG19
from keras.activations import relu
import h5py

from sklearn.metrics import calinski_harabasz_score
from keras import constraints


def _sampling_function(args):
    """
    Utility function to calculate z from the mean and log variance of a normal distribution and mask irrelevant entries.

    :param args: List of z_mean, z_log_var, and mask layers
    :return: z layer
    """
    z_mean, z_log_var, mask = args
    # assert K.shape(z_mean) == K.shape(z_log_var) == K.shape(mask), print('Wrong shape of class mask')
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return (z_mean + K.exp(0.5 * z_log_var) * epsilon) * mask


class resVAE:
    """Main class for the resVAE architecture.

    """

    def __init__(self, model_dir, config):
        """

        :param model_dir: Directory to save training weights and log files
        :param config: Config class containing parameters for resVAE
        """
        # assert os.path.isdir(model_dir)# | model_dir is None
        self.model_dir = model_dir
        self.config = config
        self.built = False
        self.compiled = False
        self.isfit = False
        self.l_rate = K.variable(0.01)
        self.genes = None
        self.classes = None
        self.dot_weights = 0
        self.gpu_count = K.tensorflow_backend._get_available_gpus()
        self.encoder, self.decoder, self.resvae_model = self.build()

    def build(self, update: bool = False):
        """
        Function that constructs the resVAE architecture, including both the encoder and decoder parts.

        :return: returns the encoder, decoder, and complete resVAE keras models
        """
        if not update:
            assert not self.built, print('Model is already built and update is set to False')
        input_shape, latent_dim = self.config['INPUT_SHAPE']
        encoder_shape = self.config['ENCODER_SHAPE']
        decoder_shape = self.config['DECODER_SHAPE']
        activ = self.config['ACTIVATION']
        last_activ = self.config['LAST_ACTIVATION']
        dropout = self.config['DROPOUT']
        latent_scale = self.config['LATENT_SCALE']
        latent_offset = self.config['LATENT_OFFSET']
        decoder_bias = self.config['DECODER_BIAS']
        decoder_regularizer = self.config['DECODER_REGULARIZER']
        decoder_regularizer_initial = self.config['DECODER_REGULARIZER_INITIAL']
        base_loss = self.config['BASE_LOSS']
        decoder_bn = self.config['DECODER_BN']
        relu_thresh = self.config['DECODER_RELU_THRESH']
        assert activ in ['relu', 'elu'], print('invalid activation function')
        assert last_activ in ['sigmoid', 'softmax', 'relu', None], print('invalid final activation function')
        assert decoder_bias in ['all', 'none', 'last']
        assert decoder_regularizer in ['var_l1', 'var_l2', 'var_l1_l2',
                                       'l1', 'l2', 'l1_l2', 'dot', 'dot_weights', 'none']
        assert base_loss in ['mse','mae']

        if decoder_regularizer == 'dot_weights':
            self.dot_weights = np.zeros(shape=(latent_scale*latent_dim, latent_scale*latent_dim))
            for s in range(latent_dim):
                self.dot_weights[s*latent_scale:s*latent_scale+latent_scale,
                s*latent_scale:s*latent_scale+latent_scale] = 1

        # L1 regularizer with the scaling factor updateable through the l_rate variable (callback)
        def variable_l1(weight_matrix):
            return self.l_rate * K.sum(K.abs(weight_matrix))

        # L2 regularizer with the scaling factor updateable through the l_rate variable (callback)
        def variable_l2(weight_matrix):
            return self.l_rate * K.sum(K.square(weight_matrix))

        # Mixed L1 and L2 regularizer, updateable scaling. TODO: Consider implementing different scaling factors for L1 and L2 part
        def variable_l1_l2(weight_matrix):
            return self.l_rate * (K.sum(K.abs(weight_matrix)) + K.sum(K.square(weight_matrix))) * 0.5

        # Dot product-based regularizer
        def dotprod_weights(weights_matrix):
            penalty_dot = self.l_rate * K.mean(K.square(K.dot(weights_matrix,
                                                              K.transpose(weights_matrix)) * self.dot_weights))
            penalty_l1 = 0.000 * self.l_rate * K.sum(K.abs(weights_matrix))
            return penalty_dot + penalty_l1

        def dotprod(weights_matrix):
            penalty_dot = self.l_rate * K.mean(K.square(K.dot(weights_matrix, K.transpose(weights_matrix))))
            penalty_l1 = 0.000 * self.l_rate * K.sum(K.abs(weights_matrix))
            return penalty_dot + penalty_l1

        def dotprod_inverse(weights_matrix):
            penalty_dot = 0.1 * K.mean(K.square(K.dot(K.transpose(weights_matrix), weights_matrix) * self.dot_weights))
            penalty_l1 = 0.000 * self.l_rate * K.sum(K.abs(weights_matrix))
            return penalty_dot + penalty_l1

        def relu_advanced(x):
            return K.relu(x, threshold=relu_thresh)

        if activ == 'relu':
            activ = relu_advanced

        # assigns the regularizer to the scaling factor. TODO: Look for more elegant method
        if decoder_regularizer == 'var_l1':
            reg = variable_l1
            reg1 = variable_l1
        elif decoder_regularizer == 'var_l2':
            reg = variable_l2
            reg1 = variable_l2
        elif decoder_regularizer == 'var_l1_l2':
            reg = variable_l1_l2
            reg1 = variable_l1_l2
        elif decoder_regularizer == 'l1':
            reg = regularizers.l1(decoder_regularizer_initial)
            reg1 = regularizers.l1(decoder_regularizer_initial)
        elif decoder_regularizer == 'l2':
            reg = regularizers.l2(decoder_regularizer_initial)
            reg1 = regularizers.l2(decoder_regularizer_initial)
        elif decoder_regularizer == 'l1_l2':
            reg = regularizers.l1_l2(l1=decoder_regularizer_initial, l2=decoder_regularizer_initial)
            reg1 = regularizers.l1_l2(l1=decoder_regularizer_initial, l2=decoder_regularizer_initial)
        elif decoder_regularizer == 'dot':
            reg = dotprod
            reg1 = dotprod
        elif decoder_regularizer == 'dot_weights':
            reg1 = dotprod_weights
            reg = dotprod
        else:
            reg = None
            reg1 = None
        resvae_inp = layers.Input(shape=(input_shape,),
                               name='Input')
        resvae_inp_cat = layers.Input(shape=(latent_dim,),
                                   name='Category_input')
        x = layers.Dense(encoder_shape[0],
                         activation=activ,
                         name='Dense1')(resvae_inp)
        x = layers.Dropout(dropout,
                           name='Dropout1')(x)
        # add layers according to encoder shape input. TODO: Consider allowing different parameters for each layer besides size.
        if len(encoder_shape) > 1:
            for i in range(len(encoder_shape)-1):
                x = layers.Dense(encoder_shape[i+1],
                                 activation=activ,
                                 name='Dense'+str(i+2))(x)
                x = layers.Dropout(dropout,
                                   name='Dropout'+str(i+2))(x)
        resvae_z_mean = layers.Dense(latent_dim * latent_scale,
                                  name='z_mean',
                                  activity_regularizer=None)(x)
        resvae_z_log_var = layers.Dense(latent_dim * latent_scale,
                                     name='z_log_var')(x)
        resvae_repeat_cat = layers.RepeatVector(latent_scale)(resvae_inp_cat)
        resvae_repeat_flattened = layers.Flatten(data_format='channels_first',
                                              name='Flatten')(resvae_repeat_cat)
        resvae_z = layers.Lambda(_sampling_function,
                              output_shape=(latent_dim * latent_scale,),
                              name='z')([resvae_z_mean, resvae_z_log_var, resvae_repeat_flattened])
        resvae_encoder = Model([resvae_inp, resvae_inp_cat],
                            [resvae_z_mean, resvae_z_log_var, resvae_z],
                            name='encoder')
        resvae_latent_inputs = layers.Input(shape=(latent_dim * latent_scale,),
                                         name='z_sampling')
        d = layers.Dense(decoder_shape[0],
                         activation=activ,
                         name='Dense_D1',
                         activity_regularizer=reg1)(resvae_latent_inputs)
        if decoder_bn:
            d = layers.BatchNormalization()(d)
        # adds layers to the decoder. See encoder layers
        if len(decoder_shape) > 1:
            for i in range(len(decoder_shape)-1):
                if decoder_bias == 'all':
                    d = layers.Dense(decoder_shape[i+1],
                                     activation=activ,
                                     name='Dense_D'+str(i+2),
                                     use_bias=True,
                                     activity_regularizer=reg)(d)
                else:
                    d = layers.Dense(decoder_shape[i+1],
                                     activation=activ,
                                     name='Dense_D' + str(i + 2),
                                     use_bias=False,
                                     kernel_regularizer=reg)(d)
                if decoder_bn:
                    d = layers.BatchNormalization()(d)
        if decoder_bias == 'none':
            resvae_outputs = layers.Dense(input_shape,
                                       activation=last_activ,
                                       use_bias=False)(d)
        else:
            resvae_outputs = layers.Dense(input_shape,
                                       activation=last_activ)(d)
        resvae_decoder = Model(resvae_latent_inputs,
                            resvae_outputs,
                            name='decoder')
        outputs = resvae_decoder(resvae_encoder([resvae_inp, resvae_inp_cat])[2])
        resvae = Model([resvae_inp, resvae_inp_cat],
                    outputs,
                    name='resvae')
        # Add a loss that is a mixture of the mean-squared error and the KL-divergence from a Gaussian of the latent space
        if base_loss == 'mse':
            reconstruction_loss = losses.mse(resvae_inp, outputs)
        else:
            reconstruction_loss = losses.mae(resvae_inp, outputs)
        reconstruction_loss *= input_shape
        # Calculate Kullback-Leibler divergence for the latent space. This is modified by adding a variable shifting the mean from zero.
        kl_loss = (1 + resvae_z_log_var - K.square(latent_offset - resvae_z_mean) - K.exp(resvae_z_log_var)) * \
                  resvae_repeat_flattened
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae_loss = vae_loss
        resvae.add_loss(vae_loss)
        # set built to true to later avoid inadvertently overwriting a built model. TODO: implement this check
        self.built = True
        return resvae_encoder, resvae_decoder, resvae

    def loss(self, y_true=None, y_pred=None):
        return self.vae_loss

    def compile(self):
        """
        This function compiles the keras model of the complete resVAE network.

        :return: None
        """
        # TODO: Add metrics
        if self.config['MULTI_GPU'] == True:
            self.encoder = multi_gpu_model(model=self.encoder, gpus=len(self.gpu_count))
            self.decoder = multi_gpu_model(model=self.decoder, gpus=len(self.gpu_count))
            self.resvae_model = multi_gpu_model(model=self.resvae_model, gpus=len(self.gpu_count))
        elif type(self.config['MULTI_GPU']) == int:
            assert self.config['MULTI_GPU'] > 0, print('Please choose an integer larger than 0 for the GPU count.')
            assert self.config['MULTI_GPU'] <= len(self.gpu_count), \
                print('Number of GPU counts provided ('+ str(self.config['MULTI_GPU'])+') may not exceed available GPU count ('+str(len(self.gpu_count))+').')
            self.encoder = multi_gpu_model(model=self.encoder, gpus=self.config['MULTI_GPU'])
            self.decoder = multi_gpu_model(model=self.decoder, gpus=self.config['MULTI_GPU'])
            self.resvae_model = multi_gpu_model(model=self.resvae_model, gpus=self.config['MULTI_GPU'])
        optimizer = self.config['OPTIMIZER']
        self.resvae_model.compile(optimizer=optimizer)
        self.compiled = True

    def fit(self, exprs=None, classes=None, model_dir=None, model_name='my_rlvae'):
        """
        Function to fit the complete resVAE model to the provided training data.

        :param exprs: Input matrix (samples x features)
        :param classes: One-hot encoded or partial class identity matrix (samples x classes)
        :param model_dir: Directory to save training logs in
        :param model_name: Name of the model (for log files
        :return: Returns a keras history object and a dictionary with scores
        """
        assert self.compiled, print('Please compile the model first by running rlvae.compile()')
        batch_size = self.config['BATCH_SIZE']
        epochs = self.config['EPOCHS']
        steps_per_epoch = self.config['STEPS_PER_EPOCH']
        validation_split = self.config['VALIDATION_SPLIT']
        callback = []
        if self.config['CB_LR_USE']:
            callback.append(callbacks.ReduceLROnPlateau(monitor=self.config['CB_MONITOR'],
                                                        factor=self.config['CB_LR_FACTOR'],
                                                        patience=self.config['CB_LR_PATIENCE'],
                                                        min_delta=self.config['CB_LR_MIN_DELTA'],
                                                        verbose=True))
        if self.config['CB_ES_USE']:
            callback.append(callbacks.EarlyStopping(monitor=self.config['CB_MONITOR'],
                                                    patience=self.config['CB_ES_PATIENCE'],
                                                    min_delta=self.config['CB_ES_MIN_DELTA'],
                                                    verbose=True))
        if model_dir is not None:
            callback.append(callbacks.CSVLogger(os.path.join(model_dir, str(model_name + '.log'))))
        decoder_regularizer = self.config['DECODER_REGULARIZER']
        decoder_regularizer_initial = self.config['DECODER_REGULARIZER_INITIAL']
        if decoder_regularizer in ['var_l1', 'var_l2', 'var_l1_l2']:
            callback.append(callbacks.LambdaCallback(on_epoch_end=lambda epoch,
                                                      logs: K.set_value(self.l_rate,
                                                                        decoder_regularizer_initial * (epoch + 1))))
        history = self.resvae_model.fit([exprs, classes],
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_split=validation_split,
                                        callbacks=callback)
        self.isfit = True
        fpc_real = self.calc_fpc()
        scores = {'fpc_real': fpc_real}
        return history, scores

    def smooth(self, exprs, classes):
        """
        This generates smoothed values, i.e. values output by the complete resVAE model.

        :param exprs: Matrix with values to be smoothed (samples x features)
        :param classes: One-hot encoded or partial class identity matrix (samples x classes)
        :return: Smoothed value estimates
        """
        return self.resvae_model.predict([exprs, classes])

    def simulate(self, classes):
        """
        This generates simulated values based on just the decoder part of resVAE.

        :param classes: Input for the latent space (samples x (classes * latent scale factor))
        :return: Simulated values
        """
        return self.decoder.predict(classes)

    def describe_model(self, model: str = 'rlvae', plot=False, filename=None):
        """
        Generates summaries of the model architecture, prints them to stdout and optionally saves them as .png files.

        :param model: Model to show (one value or a list of 'encoder', 'decoder', or 'rlvae')
        :param plot: Boolean indicating whether to plot the model architectures as .png
        :param filename: Filenames for the plots generated. Either a single string or a list with the same length as the model list.
        :return: None
        """
        assert model in ['encoder', 'decoder', 'rlvae'], print('Unknown model identifier')
        model = np.asarray(model)
        if 'encoder' in model:
            print(self.encoder.summary())
            if plot:
                assert filename is not None, print('Please provide a file name for plotting!')
                if type(filename) == list:
                    plot_model(self.encoder, to_file=filename[np.where(model == 'encoder')[0]])
                else:
                    plot_model(self.encoder, to_file=filename)
        if 'decoder' in model:
            print(self.decoder.summary())
            if plot:
                assert filename is not None, print('Please provide a file name for plotting!')
                if type(filename) == list:
                    plot_model(self.decoder, to_file=filename[np.where(model == 'decoder')[0]])
                else:
                    plot_model(self.decoder, to_file=filename)
        if 'resvae' in model:
            print(self.resvae_model.summary())
            if plot:
                assert filename is not None, print('Please provide a file name for plotting!')
                if type(filename) == list:
                    plot_model(self.resvae_model, to_file=filename[np.where(model == 'rlvae')[0]])
                else:
                    plot_model(self.resvae_model, to_file=filename)
    
    def add_latent_labels(self, classes):
        """
        Utility function to infer the latent variable dimension labels from the class labels and add it to the model. Labels can also be added manually trough modifying the .classes variable.
        
        :param classes: List of strings containing the class labels
        :return: None
        """
        latent_scale = self.config['LATENT_SCALE']
        self.classes = [x + '_' + str(y) for x in classes for y in range(latent_scale)]
        return None

    def load_model(self, model_file, model_weights):
        """
        Utility function to load models and their weights.

        :param model_file: File including path of the \*_resvae.\* model
        :param model_weights: File including path of the \*_resvae_weights.h5 weights
        :return: None
        """
        assert os.path.isfile(model_file), print('model file does not exist.')
        assert model_file.split('_')[-1].split('.')[0] == 'resvae', print('please use *_resvae.* model file')
        if model_file.split('.')[-1] == 'json':
            json_file = open(model_file, 'r')
            self.resvae_model = model_from_json(json_file.read())
            json_file.close()
            json_file = open(model_file.replace('_resvae.json', '_encoder.json'), 'r')
            self.encoder = model_from_json(json_file.read())
            json_file.close()
            json_file = open(model_file.replace('_resvae.json', '_decoder.json'), 'r')
            self.decoder = model_from_json(json_file.read())
            json_file.close()
            if model_weights is not None:
                assert os.path.isfile(model_weights), print('weights file does not exist')
                self.resvae_model.load_weights(model_weights)
                self.encoder.load_weights(model_weights.replace('_resvae_weights.h5', '_encoder_weights.h5'))
                self.decoder.load_weights(model_weights.replace('_resvae_weights.h5', '_decoder_weights.h5'))
        elif model_file.split('.')[-1] == 'h5':
            self.resvae_model = load_model(model_file)
            self.encoder = load_model(model_file.replace('_resvae.h5', '_encoder.json'))
            self.decoder = load_model(model_file.replace('_resvae.h5', '_decoder.json'))
        return None

    def get_latent_space(self, exprs: np.ndarray, classes: np.ndarray, cluster_index = None):
        """
        Utility function to return the activation of the latent space, given an input.

        :param exprs: Expression matrix (samples x genes)
        :param classes: One-hot encoded class matrix (samples x classes)
        :param cluster_index: The index of the cluster (according to the one-hot encoding) to be extracted. If None, returns all clusters
        :return: a numpy array of shape samples x neurons containing the neuron activations by each sample
        """
        if cluster_index is None:
            cluster_index = np.arange(self.config.INPUT_SHAPE[1])
        latent_space = self.encoder.predict([exprs, classes])[2]
        latent_space = np.reshape(latent_space, (len(exprs), self.config.INPUT_SHAPE[1], self.config.LATENT_SCALE))
        latent_space = latent_space[:, cluster_index, :]
        latent_space = latent_space.reshape((len(exprs), -1))
        if self.classes is not None:
            latent_space = pd.DataFrame(data=latent_space, columns=self.classes)
        else:
            latent_space = pd.DataFrame(data=latent_space)
        return latent_space

    def get_latent_to_gene(self, normalized: bool = False, direction: bool = True):
        """
        Extracts the weight mappings from the latent space (i.e. each cell type) to the output layer embedding the genes. Propagation of weights is realized as matrix multiplication of each intermediate layer's weight matrix.

        :param normalized: Whether to normalize the results by gene (default: False)
        :param direction: Return absolute (False) or directional (True) weights. (default: True)
        :return: Neuron-to-gene mappings for the hidden layer (pd.DataFrame)
        """
        num_weights = len(self.decoder.get_weights())
        result = self.decoder.get_weights()[0]
        for layer in range(1, num_weights, 1):
            if np.ndim(self.decoder.get_weights()[layer]) == 2:
                result = np.matmul(result, self.decoder.get_weights()[layer])
        if normalized:
            (result - result.mean(axis=0)) / result.std(axis=0)
        if self.genes is not None:
            if self.classes is not None:
                result = pd.DataFrame(data=result, columns=self.genes, index=self.classes)
            else:
                result = pd.DataFrame(data=result, columns=self.genes)
        elif self.classes is not None:
            result = pd.DataFrame(data=result, index=self.classes)
        else:
            result = pd.DataFrame(data=result)
        if direction:
            return result
        else:
            return np.abs(result)

    def get_neuron_to_gene(self, normalized: bool = False, initial_layer: int = 1, direction: bool = True):
        """
        Returns the weight mappings from intermediate decoder layer neurons to the output layer with gene embeddings. Weight propagation is realized by matrix multiplication of weight matrices.

        :param normalized: Whether to normalize the results by gene (default: False)
        :param initial_layer: Index of the decoder layer to output mappings for (default: 1)
        :param direction: Return absolute (False) or directional (True) weights. (default: True)
        :return: Neuron-to-gene mappings for a decoder layer (pd.DataFrame)
        """
        result = None
        weight_matrix_ind = np.argwhere(np.asarray([np.ndim(x) for x in self.decoder.get_weights()])==2).flatten()
        init = initial_layer
        num_weights = len(weight_matrix_ind)
        result = self.decoder.get_weights()[weight_matrix_ind[init]]
        for layer in range(init+1, num_weights, 1):
            result = np.matmul(result, self.decoder.get_weights()[weight_matrix_ind[layer]])
        if normalized:
            result = (result - result.mean(axis=0)) / result.std(axis=0)
        if self.genes is not None:
            result = pd.DataFrame(data=result, columns=self.genes)
        else:
            result = pd.DataFrame(data=result)
        if direction:
            return result
        else:
            return np.abs(result)

    def get_latent_to_neuron(self, normalized: bool = False, target_layer: int = 1, direction: bool = True):
        target = target_layer
        num_weights = len(self.decoder.get_weights())
        result = self.decoder.get_weights()[0]
        if target_layer > 1:
            for layer in range(1, num_weights, 1):
                if np.ndim(self.decoder.get_weights()[layer]) == 2:
                    target -= 1
                    result = np.matmul(result, self.decoder.get_weights()[layer])
                if target == 1:
                    break
        if self.classes is not None:
            result = pd.DataFrame(data=result, index=self.classes)
        else:
            result = pd.DataFrame(data=result)
        if normalized:
            result = (result - result.mean(axis=0)) / result.std(axis=0)
        result = pd.DataFrame(data=result)
        # TODO: fix directionality issue
        if direction:
            return result
        else:
            return np.abs(result)

    def get_gene_biases(self, relative: bool = False):
        """
        Returns the biases of the last (output layer

        :param relative: Outputs absolute biases relative to the absolute sum of inbound weight mappings
        :return: a 1D numpy ndarray of biases, length (genes)
        """
        if relative:
            return np.abs(self.decoder.get_weights()[-1]) / np.abs(np.sum(self.get_latent_to_gene(), axis=0))
        else:
            return self.decoder.get_weights()[-1]

    def save_config(self, outfile: str = 'models/config.json'):
        """
        Saves the config variable as json file.

        :param outfile: Path to the desired output file
        :return: None
        """
        if not os.path.isdir(os.path.split(outfile)[0]):
            os.mkdir(os.path.split(outfile)[0])
        dictionary = self.config
        with open(outfile, 'w') as json_file:
            json.dump(dictionary, json_file)
        return None

    def load_config(self, infile: str = 'models/config.json'):
        """
        Loads a config file stored as json.

        :param infile: Path to the .json file
        :return: None
        """
        assert(os.path.isfile(infile)), print('File does not exist')
        with open(infile) as handle:
            dictionary = json.loads(handle.read())
        self.config = dictionary
        return None

    def save_model_new(self, model_dir: str, model_name: str ='my_model'):
        """
        Utility function to save model weights and configurations.

        :param model_dir: Directory to save model in
        :param model_name: Base name of the model files (default: my_model)
        :return: None
        """
        if model_dir == None:
            model_dir = self.model_dir
        assert os.path.isdir(model_dir), print('model directory not found')
        assert self.isfit, print('Model has not been trained. To save the config, use save_config')
        self.resvae_model.save_weights(os.path.join(model_dir, str(model_name + '_resvae_weights.h5')))
        self.encoder.save_weights(os.path.join(model_dir, str(model_name + '_encoder_weights.h5')))
        self.decoder.save_weights(os.path.join(model_dir, str(model_name + '_decoder_weights.h5')))
        self.save_config(os.path.join(model_dir, str(model_name + '.config')))

    def load_model_new(self, model_dir: str, model_name: str = 'my_model'):
        """
        Utility function to load models and their weights.

        :param model_dir: Directory in which the model was saved
        :param model_name: base name of the model
        :return: None
        """
        assert os.path.isdir(model_dir), print('model directory does not exist.')
        assert os.path.isfile(os.path.join(model_dir, str(model_name + '_resvae_weights.h5'))), \
            print('model file not found')
        assert os.path.isfile(os.path.join(model_dir, str(model_name + '_encoder_weights.h5'))), \
            print('model file not found')
        assert os.path.isfile(os.path.join(model_dir, str(model_name + '_encoder_weights.h5'))), \
            print('model file not found')
        assert os.path.isfile(os.path.join(model_dir, str(model_name + '.config'))), \
            print('config file not found')
        self.load_config(os.path.join(model_dir, str(model_name + '.config')))
        self.built = False
        self.encoder, self.decoder, self.resvae_model = self.build()
        self.encoder.load_weights(os.path.join(model_dir, str(model_name + '_encoder_weights.h5')))
        self.decoder.load_weights(os.path.join(model_dir, str(model_name + '_decoder_weights.h5')))
        self.resvae_model.load_weights(os.path.join(model_dir, str(model_name + '_resvae_weights.h5')))
        self.compile()
        return None

    def calc_ch_score(self, exprs=None, normalized: bool = True, target='genes', source='latent',
                      permute=10, direction=True, index_source=1, index_target=1, use_negative=False):
        """
        Function to calculate the Calinski-Harabasz score (Variance Ratio Criterion) on the different layers.

        :param exprs: Expression matrix. Numpy array of shape samples x features
        :param normalized: Whether to normalize results by feature (default: True)
        :param target: Whether the target is 'genes' or a decoder layer 'neurons' (default: 'genes')
        :param source: The source layer, 'latent' or 'neurons' (default: 'latent')
        :param permute: How many random permutations to perform (default: 10)
        :param direction: Whether to use directionality of weights instead of absolute values (default: True)
        :param index_source: Which source layer to use with neuron to gene mappings (default: 1)
        :param index_target: Which target layer to use with latent to neuron mappings (default: 1)
        :param use_negative: Whether to use negative weights in the calculation (default: False)
        :return: two 1D numpy array with the scores, one with real scores, the second for permutations
        """
        weights = None
        if source == 'latent':
            if target == 'genes':
                weights = self.get_latent_to_gene(normalized=normalized,
                                                  direction=direction)
            else:
                weights = self.get_latent_to_neuron(normalized=normalized,
                                                    target_layer=index_target,
                                                    direction=direction)
        if source == 'neurons':
            if target == 'genes':
                weights = self.get_neuron_to_gene(normalized=normalized,
                                                  direction=direction,
                                                  initial_layer=index_source)
            else:
                print('Sorry, neuron to neuron mapping is not implemented yet')
        ch_real = []
        ch_permute = []
        for i in range(weights.shape[0]):
            cluster = i
            genes_c = np.asarray(self.genes).copy()
            pos_cutoff = cutils.calculate_elbow(weights.iloc[cluster, :])
            genes_c[weights.iloc[cluster, :] >= np.sort(weights.iloc[cluster, :])[pos_cutoff]] = 'top'
            genes_c[weights.iloc[cluster, :] < np.sort(weights.iloc[cluster, :])[pos_cutoff]] = 'bottom'
            ch_real.append(calinski_harabasz_score(exprs.T, genes_c))
            for i in range(permute):
                genes_permute = np.random.permutation(genes_c)
                ch_permute.append(calinski_harabasz_score(exprs.T, genes_permute))
            if direction == True and use_negative == True:
                neg_cutoff = cutils.calculate_elbow(weights.iloc[cluster, :], negative=True)
                genes_c[weights.iloc[cluster, :] <= np.sort(weights.iloc[cluster, :])[neg_cutoff]] = 'top'
                genes_c[weights.iloc[cluster, :] > np.sort(weights.iloc[cluster, :])[neg_cutoff]] = 'bottom'
                ch_real.append(calinski_harabasz_score(exprs.T, genes_c))
                for i in range(permute):
                    genes_permute = np.random.permutation(genes_c)
                    ch_permute.append(calinski_harabasz_score(exprs.T, genes_permute))
        ch_real = np.asarray(ch_real)
        ch_permute = np.asarray(ch_permute)
        return ch_real, ch_permute

    def calc_fpc(self, normalized: bool = True, target='genes', source='latent',
                      direction=True, index_source=1, index_target=1, use_negative=False):
        """
        Function to calculate the fuzzy partition coefficient on the different weight matrices.

        :param normalized: Whether to normalize results by feature (default: True)
        :param target: Whether the target is 'genes' or a decoder layer 'neurons' (default: 'genes')
        :param source: The source layer, 'latent' or 'neurons' (default: 'latent')
        :param direction: Whether to use directionality of weights instead of absolute values (default: True)
        :param index_source: Which source layer to use with neuron to gene mappings (default: 1)
        :param index_target: Which target layer to use with latent to neuron mappings (default: 1)
        :param use_negative: Whether to use negative weights in the calculation (default: False)
        :return: the FPC for the actual weight matrix
        """
        weights = None
        if source == 'latent':
            if target == 'genes':
                weights = self.get_latent_to_gene(normalized=normalized,
                                                  direction=direction)
            else:
                weights = self.get_latent_to_neuron(normalized=normalized,
                                                    target_layer=index_target,
                                                    direction=direction)
        if source == 'neurons':
            if target == 'genes':
                weights = self.get_neuron_to_gene(normalized=normalized,
                                                  direction=direction,
                                                  initial_layer=index_source)
            else:
                print('Sorry, neuron to neuron mapping is not implemented yet')
        if direction == True and use_negative == False:
            weights[weights < 0] = 0
        fpc = cutils.calculate_fpc(weights)
        return fpc

    def mappings_to_file(self, filename: str = 'mappings.hdf5', normalized: bool = True):
        """
        This saves the weight mappings to a hdf5 file that can later be imported into the frontend.

        :param filename: The name of the weight mappings file.
        :param normalized: Whether to save normalized weights. default: True
        :return: None
        """
        weights_clusters = self.get_latent_to_gene(normalized=normalized)
        weights_neurons1 = self.get_neuron_to_gene(normalized=normalized, initial_layer=1)
        weights_neurons2 = self.get_neuron_to_gene(normalized=normalized, initial_layer=2)
        weights_latent_neurons1 = self.get_latent_to_neuron(normalized=normalized, target_layer=1)
        weights_latent_neurons2 = self.get_latent_to_neuron(normalized=normalized, target_layer=2)
        weights_clusters.to_hdf(filename, 'weights/latent_genes', table=True, mode='a')
        weights_neurons1.to_hdf(filename, 'weights/neuron1_genes', table=True, mode='a')
        weights_neurons2.to_hdf(filename, 'weights/neuron2_genes', table=True, mode='a')
        weights_latent_neurons1.to_hdf(filename, 'weights/latent_neurons1', table=True, mode='a')
        weights_latent_neurons2.to_hdf(filename, 'weights/latent_neurons2', table=True, mode='a')
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='Path to the configuration file', type=str)
    parser.add_argument('--exprs', '-x',  help='Path to the expression matrix', type=str)
    parser.add_argument('--classes', '-y', help='Path to the class file', type=str)
    parser.add_argument('--model_dir', '-d', help='Path to model output directory', type=str)
    parser.add_argument('--model_name', '-n', help='Model base name', type=str)
    parser.add_argument('--write_weights', '-s', help='True or False; whether to write the weight mappings', type=bool)
    args = parser.parse_args()
    assert os.path.isfile(args.config), print('configuration file does not exist')
    assert os.path.isfile(args.exprs), print('expression matrix not found')
    assert os.path.isfile(args.classes), print('class file not found')
    assert os.path.isdir(args.model_dir), print('model dir does not exist')
    print('Loading expression matrix...')
    try:
        exprs, genes = cutils.load_exprs(args.exprs)
    except:
        print('Unexpected error when loading expression matrix:', sys.exc_info()[0])
        raise
    print('Normalizing expression matrix...')
    try:
        exprs_norm = cutils.normalize_count_matrix(exprs)
    except:
        print('Unexpected error when normalizing expression matrix:', sys.exc_info()[0])
        raise
    print('Loading and encoding class vector...')
    try:
        classes = np.loadtxt(args.classes)
        classes, _ = cutils.one_hot_encoder(classes)
    except:
        print('Unexpected error when loading or encoding classes:', sys.exc_info()[0])
        raise
    print('Initializing model...')
    try:
        my_resvae = resVAE(args.config, args.model_dir)
        my_resvae.compile()
        my_resvae.genes = genes
    except:
        print('Unexpected error when initializing model:', sys.exc_info()[0])
        raise
    print('Fitting model...')
    try:
        my_resvae.fit(exprs=exprs_norm, classes=classes, model_dir=args.model_dir, model_name=args.model_name)
    except:
        print('Unexpected error when fitting model:', sys.exc_info()[0])
        raise
    print('Saving model')
    try:
        my_resvae.save_model_new()
    except:
        print('Unexpected error when saving model:', sys.exc_info()[0])
        raise
    print('Extracting weights')
    try:
        weights_classes = my_resvae.get_latent_to_gene()
        weights_neurons_1 = my_resvae.get_neuron_to_gene(initial_layer=0)
        weights_classes_neurons_1 = my_resvae.get_latent_to_neuron(target_layer=0)
        if len(my_resvae.config['DECODER_SHAPE']) > 1:
            weights_neurons_2 = my_resvae.get_neuron_to_gene(initial_layer=1)
            weights_classes_neurons_2 = my_resvae.get_latent_to_neuron(target_layer=1)
    except:
        print('Unexpected error when obtaining weights:', sys.exc_info()[0])
        raise
    print('Saving weights...')
    try:
        weights_classes.to_csv(os.path.join(args.model_dir, str(args.model_name + '_weights_classes.csv')))
        weights_neurons_1.to_csv(os.path.join(args.model_dir, str(args.model_name + '_weights_neurons_1.csv')))
        weights_classes_neurons_1.to_csv(os.path.join(args.model_dir,
                                                      str(args.model_name + '_weights_classes_neurons_1.csv')))
        if len(my_resvae.config['DECODER_SHAPE']) > 1:
            weights_neurons_2.to_csv(os.path.join(args.model_dir, str(args.model_name + '_weights_neurons_2.csv')))
            weights_classes_neurons_2.to_csv(os.path.join(args.model_dir,
                                                        str(args.model_name + '_weights_classes_neurons_2.csv')))
    except:
        print('Unexpected error when saving weights:', sys.exc_info()[0])
        raise
    print('Done! resVAE did not encounter any errors.')

    return 1


if __name__ == '__main__':
    main()
