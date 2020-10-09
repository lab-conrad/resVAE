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

import codecs
import json
import os
from scipy import io, sparse

import numpy as np
import pandas as pd
import urllib3
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, Normalizer
import warnings


def one_hot_encoder_old(classes: np.ndarray):
    """
    A utility function to transform dense labels into sparse (one-hot encoded) ones. This wraps LabelEncoder from sklearn.preprocessing and to_categorical from keras.utils. If non-integer labels are supplied, the fitted LabelEncoder is returned as well.

    :param classes: A 1D numpy ndarray of length samples containing the individual samples class labels.
    :return: A 2D numpy ndarray of shape samples x classes and a None type if class labels were integers or a fitted instance of class sklearn.preprocessing.LabelEncder
    """
    warnings.warn("deprecated", DeprecationWarning)
    if classes.dtype == 'O':
        l_enc = LabelEncoder()
        classes_encoded = l_enc.fit_transform(classes)
        classes_onehot = to_categorical(classes_encoded)
        return classes_onehot, l_enc
    elif classes.dtype in ['int16', 'int32', 'int64']:
        classes_onehot = to_categorical(classes)
        return classes_onehot, None


def one_hot_encoder(classes: np.ndarray, extra_dim_reserve=False, extra_dim_num=20):
    """
    A utility function to transform dense labels into sparse (one-hot encoded) ones. This wraps LabelEncoder from sklearn.preprocessing and to_categorical from keras.utils. If non-integer labels are supplied, the fitted LabelEncoder is returned as well.
    :param classes: A 1D numpy ndarray of length samples containing the individual samples class labels.
    :return: A 2D numpy ndarray of shape samples x classes and a None type if class labels were integers or a fitted instance of class sklearn.preprocessing.LabelEncder
    """
    if extra_dim_reserve:
        classes_onehot, _ = one_hot_encoder(classes)
        classes_extra_dim = np.ones((classes_onehot.shape[0], extra_dim_num))
        classes_onehot_concat = np.concatenate((classes_extra_dim, classes_onehot), axis=1)
        return classes_onehot_concat, None
    else:
        if len(classes.shape) >= 2:
            individual_classes = np.apply_along_axis(one_hot_encoder, 0, classes)[0, 1:]
            classes_onehot_concat = np.concatenate(individual_classes, axis=1)
            return classes_onehot_concat, None
        else:
            if classes.dtype == 'O' or str(classes.dtype)[0] == '<':
                l_enc = LabelEncoder()
                classes_encoded = l_enc.fit_transform(classes)
                classes_onehot = to_categorical(classes_encoded)
                return classes_onehot, l_enc
            elif classes.dtype in ['int16', 'int32', 'int64']:
                classes_onehot = to_categorical(classes)
                return classes_onehot, None
            else:
                raise ValueError('one hot encoder can not find numpy type')


def mixed_encoder(classes: np.ndarray, extra_dim_reserve=False, extra_dim_num=20, leave_out=None):
    """
    Mixed resVAE/VAE encoder: leave unrestricted dimensions to capture unknown effects.

    :param classes: np.ndarray with class assignments.
    :param extra_dim_reserve: Whether to reserve extra (unrestricted) dimensions (default: False)
    :param extra_dim_num: Number of extra dimensions (default: 20)
    :param leave_out: Whether to leave out a certain class (default: None)
    :return: returns a one-hot encoded class matrix
    """
    if leave_out is not None:
        classes = np.delete(classes, [0], axis=1)
        left_out = classes[:, leave_out].astype(dtype=np.int)
        classes = np.delete(classes, leave_out, axis=1)
    classes_onehot, _ = one_hot_encoder(classes, extra_dim_reserve, extra_dim_num)
    if leave_out is not None:
        classes_onehot = np.concatenate((classes_onehot, left_out), axis=1)
    return classes_onehot, None


def download_gmt(url: str, destination: str or None=None, file_name: str or None=None, replace: bool = False):
    """
    Utility function to download .gmt pathway files into the correct subfolder.

    :param url: The URL of the gmt files.
    :param destination: Destination directory. If left blank, defaults to 'gmts/'
    :param file_name: The file name to write to disk. If left blank, is left unchanged from the download source files name.
    :param replace: Boolean indicating whether to overwrite if the file name already exists.
    :return: None
    """
    if destination is None:
        destination = 'gmts/'
    assert os.path.isdir(destination), print('Destination does not exist')
    if file_name is None:
        file_name = url.split('/')[-1]
    file = os.path.join(destination, file_name)
    if not replace:
        assert not os.path.exists(file), \
            print('File already exists and replace is set to False')
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    with open(file, 'wb') as out:
        while True:
            data = r.read()
            if not data:
                break
            out.write(data)
    r.release_conn()
    return None


def get_wikipathways(organism: str, destination: str or None=None, file_name: str or None=None, replace=False):
    """
    Utility function to quickly download WikiPathway data. A wrapper for download_gmt.

    :param organism: String (one of 'hs', 'mm', 'rn', 'dr', 'dm') to select download of the pathway data for either human, mouse, rat, zebra fish, or fruit fly.
    :param destination: Destination directory. If left blank, defaults to 'gmts/'
    :param file_name: The file name to write to disk. If left blank, is left unchanged from the download source files name.
    :param replace: Boolean indicating whether to overwrite if the file name already exists.
    :return: None
    """
    assert organism in ['hs', 'mm', 'rn', 'dr', 'dm'], print('Organism not found')
    if organism == 'hs':
        url = 'http://data.wikipathways.org/current/gmt/wikipathways-20190610-gmt-Homo_sapiens.gmt'
        download_gmt(url=url, destination=destination, file_name=file_name, replace=replace)
    return None


def gmt_to_json(infile: str, outfile: str or None=None):
    """
    Utility function to convert .gmt pathway files to json and write them to disk.

    :param infile: Path of the input file to convert.
    :param outfile: Output path of the corresponding .json
    :return: None
    """
    assert os.path.isfile(infile), print('Input file does not exist')
    if outfile is None:
        outfile = infile.split('.', 1)[0] + '.json'
    gmt = []
    with open(infile, 'r') as f:
        for line in f:
            gmt.append(line.strip().split('\t', 2))
    f.close()
    gmt = np.asarray(gmt)
    genes = []
    for line in gmt[:, 2].tolist():
        genes.append(line.split('\t'))
    genes = np.expand_dims(np.asarray(genes), axis=1)
    gmt = np.hstack([gmt[:, :2], genes]).tolist()
    json.dump(gmt, codecs.open(outfile, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def read_json(infile: str):
    """
    Utility function to read a .json and report the result as a list.

    :param infile: Path to the input .json.
    :return: list of gne-pathway mappings
    """
    assert os.path.isfile(infile), print('Input file does not exist')
    with open(infile) as json_file:
        data: list = json.load(json_file)
    json_file.close()    
    return data


def calculate_gmt_overlap(pathways, genelist):
    """
    Utility function to calculate the overlap between a gene set and any gene sets defined in a gmt file (nested list)

    :param pathways: A nested list such as would be obtained from running the read_json function
    :param genelist: Either a single list of genes to test, or a 2D numpy array with gene lists in columns
    :return: returns a numpy array with overlap counts
    """
    if max([len(x) for x in genelist[:][0]]) > 1:
        #genelist = np.asarray(genelist)
        hits = []
        for lst in range(len(genelist[0])):
            hits_int = []
            for i in range(len(pathways)):
                hits_int.append(len([x for x in pathways[i][2] if x in genelist[:, lst]]))
            hits.append(hits_int)
    else:
        hits = []
        for i in range(len(pathways)):
            hits.append(len([x for x in pathways[i][2] if x in genelist]))
        hits = np.asarray(hits)
    return hits


def calculate_fpc(weight_matrix: np.ndarray):
    """
    Utility function to calculate the fuzzy partition coefficient.

    :param weight_matrix: A numpy array with the neuron to gene mappings
    :return: The fuzzy partition coefficient of the weight matrix
    """
    n_genes = weight_matrix.shape[1]
    fpc = np.trace(weight_matrix.dot(weight_matrix.transpose())) / float(n_genes)
    return fpc


def normalize_count_matrix(exprs: np.ndarray):
    """
    Utility function to normalize a count matrix for the samples to sum to one. Wrapper for sklearn.preprocessing.Normalizer

    :param exprs: 2D numpy ndarray of shape samples x genes containing the gene expression values to be normalized to unit norm
    :return: a 2D numpy array of shape samples x genes of normalized expression values and a fitted instance of sklearn.preprocessing.Normalizer to be used for reconverting expression values after training resVAE
    """
    norm = Normalizer(norm='l1', copy=False)
    norm_exprs = norm.fit_transform(exprs)
    return norm_exprs, norm


def load_sparse_matrix(sparse_matrix_path):
    """
    Utility function to load a sparse matrix.

    :param sparse_matrix_path: Path to matrix file
    :return: returns a np.ndarray
    """
    file_format = '.' + sparse_matrix_path.split('.')[-1]
    assert file_format == '.mtx'
    sparse_m = io.mmread(sparse_matrix_path)
    return sparse_m.toarray()


def write_sparse_matrix(sparse_matrix, sparse_matrix_path):
    """
    Utility function to write a sparse matrix.

    :param sparse_matrix: Sparse matrix to write to file.
    :param sparse_matrix_path: File name/path to write.
    :return: None
    """
    assert sparse_matrix.dtype == 'int64'
    sparse_m = sparse.csr_matrix(sparse_matrix)
    io.mmwrite(sparse_matrix_path, sparse_m)
    return None


def compose_dataframe(array, index, columns):
    """
    Utility function to convert a numpy array to a pandas DataFrame.

    :param array: the np.ndarray to convert
    :param index: row names
    :param columns: column names
    :return: a pandas DataFrame
    """
    assert len(array.shape) == 2
    assert array.shape[0] == len(index) and array.shape[1] == len(columns)
    return pd.DataFrame(array, index=index, columns=columns)


def decompose_dataframe(df):
    """
    Utility function to decompose a pandas DataFrame to numpy ndarrays.

    :param df: a pandas DataFrame
    :return: three np.ndarrays with the data, row names, and column names
    """
    index, column, array = df.index, df.column, df.values
    return array, index, column


def load_txt_file(txt_file_path):
    """
    Utility function to load a text file as list.

    :param txt_file_path: The path to the file (must be .txt)
    :return: The file content in list format
    """
    file_format = '.' + txt_file_path.split('.')[-1]
    assert file_format == '.txt'
    txt_file_array = np.loadtxt(txt_file_path, dtype=np.str)
    return txt_file_array.tolist()


def write_txt_file(txt_file_path, array):
    """
    Utility function to write a txt file.

    :param txt_file_path: The path and filename to write at
    :param array: The np.ndarray to write to file
    :return: None
    """
    np.savetxt(txt_file_path, array, delimiter=',')
    return None


def load_sparse(sparse_matrix_path, index_txt_file_path, column_txt_file_path):
    """
    Utility function to load a sparse matrix with row and column names as txt files.

    :param sparse_matrix_path: Path to sparse matrix
    :param index_txt_file_path: Path to row names
    :param column_txt_file_path: Path to column names
    :return: A DataFrame of a sparse matrix with row and column names
    """
    sparse_matrix = load_sparse_matrix(sparse_matrix_path)
    index, column = load_txt_file(index_txt_file_path), load_txt_file(column_txt_file_path)
    return compose_dataframe(sparse_matrix, index, column)


def write_sparse(df, sparse_matrix_path, index_txt_file_path, column_txt_file_path):
    """
    Utility function to write a DataFrame as sparse matrix, with row and column names.

    :param df: The DataFrame to write to file
    :param sparse_matrix_path: File path to write the sparse matrix to
    :param index_txt_file_path: row name file path
    :param column_txt_file_path: column name file path
    :return: None
    """
    sparse_matrix, index, column = decompose_dataframe(df)
    write_sparse_matrix(sparse_matrix, sparse_matrix_path)
    index = list(index)
    write_txt_file(index_txt_file_path, index)
    column = list(column)
    write_txt_file(column_txt_file_path, column)
    return None


def load_exprs(path, sep: str = ',', order: str = 'cg', sparse=None):
    """
    Utility function to load expression matrices, extract gene names, and return both
    :param path: Path to the expression matrix
    :param sep: Separator used in the expression file (default: ',')
    :return: a 2D numpy ndarray with the expression values and a pandas index object containing the ordered gene names
    """
    if sparse:
        sparse_matrix_path, index_txt_file_path, column_txt_file_path = (sparse['sparse_matrix_path'],
                                                                         sparse['index_txt_file_path'],
                                                                         sparse['column_txt_file_path'])
        exprs = load_sparse(sparse_matrix_path, index_txt_file_path, column_txt_file_path)
    else:
        # TODO: Update Docstring
        assert os.path.isfile(path), print('Invalid file path')
        assert order in ['cg', 'gc']
        ext = path.split('.')[-1]
        assert ext in ['csv', 'tsv', 'pkl', 'feather'], print('Unrecognized file format. Currently supported formats include: csv, tsv, pkl and feather.')
        if ext == 'csv':
            exprs = pd.read_csv(path, sep=sep, header=0, index_col=0)
        elif ext == 'tsv':
            exprs = pd.read_csv(path, sep='\t', header=0, index_col=0)
        elif ext == 'pkl':
            exprs = pd.read_pickle(path)
        elif ext == 'feather':
            exprs = pd.read_feather(path)
        if not order == 'cg':
            exprs = exprs.T
    return np.asarray(exprs), exprs.columns


def calculate_elbow(weights: np.ndarray, negative: bool = False):
    """
    Calculates the position of the elbow point for each weight matrix.

    :param weights: A 1D or 2D numpy ndarray of length genes or shape neurons x genes containing weight mappings
    :param negative: Boolean indicating whether to return negatively enriched indices (default: False)
    :return: Returns an integer (1D input) or 1D numpy ndarray (2D input) with the position of the elbow point along a sorted axis
    """
    if weights.ndim == 1:
        if negative:
            weights_current = np.sort(np.abs(weights[weights < 0] / np.min(weights[weights < 0])))
            weights_index = np.arange(len(weights_current)) / np.max(len(weights_current))
            distance = weights_index - weights_current
            return len(weights_index) - np.argmax(distance)
        else:
            weights_current = np.sort(np.abs(weights[weights >= 0] / np.max(weights[weights >= 0])))
            weights_index = np.arange(len(weights_current)) / np.max(len(weights_current))
            distance = weights_index - weights_current
            return np.argmax(distance) + np.sum(weights < 0)
    if weights.ndim > 1:
        distances = []
        if negative:
            weights_index = np.arange(weights.shape[1]) / np.min(weights.shape[1])
            for neuron in range(weights.shape[0]):
                weights_current = np.sort(np.abs(weights[neuron, weights < 0] / np.max(weights[neuron, weights < 0])))
                distance = weights_index - weights_current
                distances.append(len(weights_index) - np.argmax(distance))
        else:
            weights_index = np.arange(weights.shape[1]) / np.max(weights.shape[1])
            for neuron in range(weights.shape[0]):
                weights_current = np.sort(np.abs(weights[neuron, weights < 0] / np.max(weights[neuron, weights < 0])))
                distance = weights_index - weights_current
                distances.append(np.argmax(distance) + np.sum(weights < 0))
        return distances


def assert_config(config: dict):
    assert len(config['INPUT_SHAPE']) == 2, print('Input shape has wrong dimensionality')
    assert type(config['ENCODER_SHAPE']) == list, \
        print('Encoder shape is not a list')
    assert len(config['ENCODER_SHAPE']) >= 1, \
        print('Missing encoder dimensions')
    assert type(config['DECODER_SHAPE']) == list,\
        print('Decoder shape is not a list')
    assert len(config['DECODER_SHAPE']) >= 1, \
        print('Missing decoder dimensions')
    assert config['ACTIVATION'] in ['relu', 'elu', 'sigmoid', 'tanh', 'softmax', 'selu'], \
        print('Unknown hidden layer activation function')
    assert config['LAST_ACTIVATION'] in ['relu', 'elu', 'sigmoid', 'tanh', 'softmax', 'selu', None], \
        print('Unknown final layer activation function')
    assert type(config['DROPOUT']) in [int, type(None), float], print('Invalid value for dropout')
    if config['DROPOUT']:
        assert config['DROPOUT'] < 1, print('Dropout too high')
    assert type(config['LATENT_SCALE']) == int and config['LATENT_SCALE'] >= 1, \
        print('Invalid value for latent scale. Please choose an integer larger than or equal to one')
    assert type(config['BATCH_SIZE']) == int and config['BATCH_SIZE'] >= 1, \
        print('Invalid value for batch size. Please choose an integer larger than or equal to one')
    assert type(config['EPOCHS']) == int and config['EPOCHS'] >= 1, \
        print('Invalid value for epochs. Please choose an integer larger than or equal to one')
    assert type(config['STEPS_PER_EPOCH']) in [type(None), int], \
        print('Invalid value for steps per epoch. Please choose None or an integer larger than or equal to one')
    assert type(config['VALIDATION_SPLIT']) in [float, type(None)], \
        print('Invalid value for validation split. Please choose None or a float value smaller than one')
    assert type(config['LATENT_OFFSET']) in [float, int], \
        print('Please choose a number for the latent offset')
    assert config['DECODER_BIAS'] in ['last', 'all', 'none'], \
        print('Invalid value for decoder bias. Please choose all, none, or last')
    assert config['DECODER_REGULARIZER'] in ['none', 'l1', 'l2', 'l1_l2',
                                             'var_l1', 'var_l2', 'var_l1_l2', 'dot', 'dot_weights'], \
        print('Invalid value for decoder regularizer. Please choose one of '
              'none, l1, l2, l1_l2, var_l1, var_l2, or var_l1_l2')
    if config['DECODER_REGULARIZER'] != 'none':
        assert type(config['DECODER_REGULARIZER_INITIAL']) == float, \
            print('Please choose a float value as (initial) decoder regularizer penalty')
    assert config['BASE_LOSS'] in ['mse', 'mae'], \
        print('Please choose mse or mae as base loss')
    assert type(config['DECODER_BN']) == bool, \
        print('Please choose True or False for the decoder batch normalization')
    assert type(config['CB_LR_USE']) == bool, \
        print('Please choose True or False for the learning rate reduction on plateau')
    assert type(config['CB_ES_USE']) == bool, \
        print('Please choose True or False for the early stopping callback')
    if config['CB_LR_USE']:
        assert type(config['CB_LR_FACTOR']) == float, \
            print('Please choose a decimal value for the learning rate reduction factor')
        assert type(config['CB_LR_PATIENCE']) == int and config['CB_LR_PATIENCE'] >= 1, \
            print('Please choose an integer value equal to or larger than one for the learning rate reduction patience')
        assert type(config['CB_LR_MIN_DELTA']) == float or config['CB_LR_MIN_DELTA'] == 0, \
            print('Please choose a floating point value or 0 for the learning rate reduction minimum delta')
    if config['CB_ES_USE']:
        assert type(config['CB_ES_PATIENCE']) == int and config['CB_ES_PATIENCE'] >= 1, \
            print('Please choose an integer value equal to or larger than one for the early stopping patience')
        assert type(config['CB_ES_MIN_DELTA']) == float or config['CB_ES_MIN_DELTA'] == 0, \
            print('Please choose a floating point value or 0 for the early stopping minimum delta')
    if config['CB_LR_USE'] or config['CB_ES_USE']:
        assert config['CB_MONITOR'] in ['loss', 'val_loss'], \
            print('Please choose loss or val_loss as metric to monitor for callbacks')
    assert type(config['MULTI_GPU']) in ['bool', 'int']
