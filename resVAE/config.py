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

config = {'INPUT_SHAPE': (450, 10),
          'ENCODER_SHAPE': [512, 256],
          'DECODER_SHAPE': [256, 512],
          'ACTIVATION': 'relu',
          'LAST_ACTIVATION': 'softmax',
          'DROPOUT': 0,
          'LATENT_SCALE': 5,
          'OPTIMIZER': 'Adam',
          'BATCH_SIZE': 64,
          'EPOCHS': 300,
          'STEPS_PER_EPOCH': None,
          'VALIDATION_SPLIT': 0.2,
          'VALIDATION_STEPS': 10,
          'LATENT_OFFSET': 10,
          'DECODER_BIAS': 'last',
          'DECODER_REGULARIZER': 'var_l1',
          'DECODER_REGULARIZER_INITIAL': 0.0001,
          'DECODER_RELU_THRESH': 0,
          'BASE_LOSS': 'mse',
          'DECODER_BN': False,
          'CB_MONITOR': 'val_loss',
          'CB_LR_USE': True,
          'CB_LR_FACTOR': 0.2,
          'CB_LR_PATIENCE': 15,
          'CB_LR_MIN_DELTA': 1e-8,
          'CB_ES_USE': True,
          'CB_ES_PATIENCE': 30,
          'CB_ES_MIN_DELTA': 0,
          'MULTI_GPU': False
          }
