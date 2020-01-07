
# coding: utf-8

# In[ ]:


import segmentation_models as sm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_int, img_as_ubyte
from skimage.filters import threshold_otsu
from keras import backend as K

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from PIL import Image
from scipy.ndimage.morphology import binary_fill_holes
import utils
from utils import Dataloder, Dataset

    # train model
def train(model,train_dataloader,valid_dataloader,epoch,model_dir):
    history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=epoch, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),)
    model.save_weights(model_dir)
    
def predict(model,x_test_dir,y_test_dir,model_dir):
    model.load_weights(model_dir)
    names=os.listdir(x_test_dir)
  
    for i in range(len(names)):
        image=cv2.imread(os.path.join(x_test_dir,names[i]))[:192,:192]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).squeeze()
        pr_mask=img_as_ubyte(pr_mask<0.5)

        io.imsave(os.path.join(y_test_dir,names[i]),pr_mask)


            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='passing')

    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'predict'")
    parser.add_argument('--epoch', type=int,required=False,)
    parser.add_argument('--BATCH_SIZE', type=int,required=False,)
    parser.add_argument('--LR', type=float,required=False,)


    parser.add_argument('--train_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the training dataset')


    parser.add_argument('--valid_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the validation dataset')


    parser.add_argument('--test_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the test dataset')
    

    parser.add_argument('--model_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the training weight')

    args = parser.parse_args()      
  
    x_train_dir = os.path.join(args.train_dir,'input')
    y_train_dir = os.path.join(args.train_dir,'output')

    x_valid_dir = os.path.join(args.valid_dir,'input')
    y_valid_dir = os.path.join(args.valid_dir,'output')

    x_test_dir = os.path.join(args.test_dir,'input')
    y_test_dir = os.path.join(args.test_dir,'output')


    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    LR = 0.07
    LR=args.LR
    optim = keras.optimizers.SGD(LR)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


    CLASSES=['pore']

    BATCH_SIZE = 8
    BATCH_SIZE = args.BATCH_SIZE

    EPOCHS = 10


    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        classes=CLASSES, class_values=[0], row=192,column=192,ini_val=0
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES, class_values=[0], row=192,column=192,ini_val=0
    )

    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        classes=CLASSES,  class_values=[0],  row=192,column=192,ini_val=0
    )


    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)




    #keras.backend.set_image_data_format('channels_last')
    keras.backend.set_image_data_format('channels_last')

    model = sm.Unet(BACKBONE, encoder_weights='imagenet',classes=1,input_shape=(192, 192, 3),activation='sigmoid')

    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model_1.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]


    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() 
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    #optim = keras.optimizers.Adam(LR)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5),'accuracy']

    model.compile(
        optim,
        loss=keras.losses.binary_crossentropy,
        metrics=metrics,
    )
    if args.command == "train":
        train(model,train_dataloader,valid_dataloader,args.epoch,args.model_dir)
    elif args.command == "predict":
        predict(model,x_test_dir,y_test_dir,args.model_dir)



# %%
