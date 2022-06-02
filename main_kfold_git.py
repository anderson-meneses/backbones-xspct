import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

#import sys
import glob
import time
import random
from PIL import Image

from unet import unet_completa, dice_coef
from unet_hedden import unet_hedden
from utils import create_folder, load_images_array
from sklearn.model_selection import train_test_split
from utils_metrics import jaccard_coef, dice_coef_2, calcula_metricas
from utils_img import save_images

from segmentation_models import Unet, Linknet, FPN, PSPNet
from segmentation_models.losses import bce_jaccard_loss, DiceLoss
from segmentation_models.metrics import iou_score

# Data augmentation 1 - 2022.05.07 Acrescentando DA no conj de valid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import Adam

trainAug = Sequential([
	preprocessing.RandomFlip("horizontal"),
	preprocessing.RandomZoom(
		height_factor=(-0.2, +0.2),
		width_factor=(-0.2, +0.2)),
	preprocessing.RandomRotation(0.1)
])

valAug = Sequential([
	preprocessing.RandomFlip("horizontal"),
	preprocessing.RandomZoom(
		height_factor=(-0.2, +0.2),
		width_factor=(-0.2, +0.2)),
	preprocessing.RandomRotation(0.1)
])

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    validation_split=0.1)

image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
ORIGINAL_SIZE = 850
NEW_SIZE = 256

# Choose train folder TM40 ou TM46
_folder = './dados_girino/TM40_46prod'
# _folder = './dados_girino/TM46_40prod'

norm_imgs = sorted(glob.glob(_folder + '/A1_norm_images/*'))
GT_imgs = sorted(glob.glob(_folder + '/A2_GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X = load_images_array(norm_imgs, new_size = NEW_SIZE)
Y = load_images_array(GT_imgs, new_size = NEW_SIZE)


print(X.shape)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

use_batch_size = 4

epochs = 100 

create_folder('outputs')

n_exec = 1
n_fold = 2 #11 #2  

iou_cv, dice_cv = [], [] # lista de listas desta execucao de cv
 
for i in range(n_fold):
    
    TEMPO = []
    time_train_1 = time.time()

    random.seed(time.time())
    seed_min = 0
    seed_max = 2**20
    SEED_1 = random.randint(seed_min, seed_max)
    SEED_2 = random.randint(seed_min, seed_max)
    SEED_3 = random.randint(seed_min, seed_max)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED_1)

    # Data Augmentation 2 - 2022.05.07 Fazendo DA no conj de valid
    trainDS = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    trainDS = trainDS.repeat(3)
    trainDS = (
     	trainDS
     	.shuffle(use_batch_size * 100)
     	.batch(use_batch_size)
     	.map(lambda x, y: (trainAug(x), trainAug(y)), num_parallel_calls=AUTOTUNE)
     	.prefetch(tf.data.AUTOTUNE)
     )

    valDS = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    valDS = valDS.repeat(3)
    valDS = (
     	valDS
     	.shuffle(use_batch_size * 100)
     	.batch(use_batch_size)
     	.map(lambda x, y: (valAug(x), valAug(y)), num_parallel_calls=AUTOTUNE)
     	.prefetch(tf.data.AUTOTUNE)
     )
     
    N = X_train.shape[-1]

    # Models 
 
    # Unet effnet 
    # model = Unet(backbone_name='efficientnetb0', encoder_weights=None,
    #               input_shape=(None,None,N))
       
    # Linknet resnet34 
    model = Linknet(backbone_name='resnet34', encoder_weights=None,
                  input_shape=(None,None,N))

    model.compile(optimizer=Adam(), loss=bce_jaccard_loss, metrics=[iou_score]) 
    
    history = model.fit(trainDS, 
              epochs=epochs, callbacks=callback, 
              validation_data=valDS)

    folder_name = './outputs/Exec_%s'%str(n_exec)
    create_folder(folder_name)
    name_file = str(use_batch_size) + "_" + str(epochs) + "_exec_%i"%n_exec + "_fold_%i"%i
    model.save(folder_name + '/girino_%s.h5'%name_file)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(folder_name + '/loss_%i.png'%i)
    plt.close()
    np.save(folder_name + '/history_%i.npy'%i, history.history)

    print("Calculando o dice para as imagens de teste")
  
    predicao = np.float64(model.predict(X_val))
    