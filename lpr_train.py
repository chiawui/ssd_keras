
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
#from tensorflow.keras.models import load_model
from math import ceil
import numpy as np

#from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras.regularizers import l2
from keras.models import load_model

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.models import Model
import utils

########################################################################
# configuration
#tf.enable_eager_execution()
input_shape=(224,224,3)
classes=['bg','1','2','3']
class_count=len(classes)-1 #-1 for background class
feature_map_shape=(3,3)
n_boxes=1 #no. of box within 1 unit of feature map
clip_boxes=False
variances=[0.1, 0.1, 0.2, 0.2]
########################################################################

train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

dir=['D:/Exercise/research/test_lpr_v3']
filenames=['D:/Exercise/research/test_lpr_v3/filenames.txt']
val_filesnames=['D:/Exercise/research/test_lpr_v3/validation.txt']
train_dataset.parse_xml(dir, filenames, dir, classes=classes)
val_dataset.parse_xml(dir, val_filesnames, dir, classes=classes)

#train_dataset.create_hdf5_dataset()

########################################################################
# build model

model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
x=model.output
x = MaxPooling2D(pool_size=(2, 2), name='pool6')(x)
classes4 = Conv2D(n_boxes * (class_count+1), feature_map_shape, strides=(1, 1), padding="same", kernel_initializer='he_normal', name='classes')(x)
boxes4 = Conv2D(n_boxes * 4, feature_map_shape, strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), name='boxes4')(x)
anchors4 = AnchorBoxes(input_shape[0], input_shape[1], this_scale=0.33, next_scale=0.33, aspect_ratios=[1.0], variances=variances,
                           two_boxes_for_ar1=False, normalize_coords=True, clip_boxes=clip_boxes, name='anchors4')(boxes4)
classes4_reshaped = Reshape((-1, class_count+1), name='classes4_reshape')(classes4)
boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)

classes_softmax = Activation('softmax', name='classes_softmax')(classes4_reshaped)
predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes4_reshaped, anchors4_reshaped])

for layer in model.layers:
    layer.trainable = False
ext_model = Model(inputs=model.input, outputs=predictions)
ext_model.summary()

# 2: Optional: Load some weights
#model.load_weights('./ssd7_weights.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

ext_model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

########################################################################
# create encoder and generator
batch_size=5
encoder = SSDInputEncoder(input_shape[0], input_shape[1], class_count, clip_boxes=clip_boxes, two_boxes_for_ar1=False, 
                          aspect_ratios_global=[1.0], predictor_sizes=[feature_map_shape], min_scale=0.33, max_scale=0.33, pos_iou_threshold=0.7,
                          variances=variances)

data_augmentation_chain = DataAugmentationConstantInputSize(random_flip=0, random_scale=(0.8, 1.01,0.5), clip_boxes=clip_boxes)
generator=train_dataset.generate(batch_size, label_encoder=encoder, degenerate_box_handling='warn', transformations=[data_augmentation_chain])
val_generator=val_dataset.generate(batch_size, label_encoder=encoder, degenerate_box_handling='warn', transformations=[])

########################################################################
# setup training
# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 80
final_epoch     = 100
steps_per_epoch = 20

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=4,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.000001)

tensorboard = TensorBoard(log_dir='./logs/val_loss', 
                          histogram_freq=0, batch_size=32, 
                          write_graph=True, write_grads=False, 
                          write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, 
                          update_freq='epoch')

ext_model.load_weights('lpr_train.h5', by_name=True)
history = ext_model.fit_generator(generator=generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=[reduce_learning_rate, tensorboard],
                              #validation_data=val_generator,
                              #validation_steps=1,
                              initial_epoch=initial_epoch)

ext_model.save_weights('lpr_train.h5')
########################################################################
# diagnostic
#with tf.Session() as sess:
#img, label=next(generator)
#utils.save_img(img[0], 'test.png')
#print(np.argmax(label[0,:,0:4], axis=1))
#print(np.reshape(np.argmax(label[0,:,0:4], axis=1), (3,3)))
#print(label[0][14][:])

