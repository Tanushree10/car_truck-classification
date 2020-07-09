from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Dense,Activation,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import h5py
import numpy
import cv2
import tensorflow
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
classifier = load_model('Project1.h5')
img1 = image.load_img('Data/test/truck/68548902.jpg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
if(prediction[:,:]>0.5):
    value ='Truck :%1.2f'%(prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value ='Car :%1.2f'%(1.0-prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

plt.imshow(img1)
plt.show()

