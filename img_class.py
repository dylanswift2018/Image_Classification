#importing the dataset 
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt 

#loading the training data and testing data :) 
(train_x,train_y),(test_x,test_y)= cifar10.load_data()

#plotting some images to visualize the dataset 
#n=6
#plt.figure(figsize=(20,10))
#for x in range (n) :
#  plt.subplot(330+1+x)
#  plt.imshow(train_x[x])
#  plt.show()

#importing the required layers and modules to create our convultional neural netwrok architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils
#import np_utils

#converting the pixel values of the dataset to float type and normalizing the dataset 
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x = train_x/255.0
test_x = test_x/255.0

#encoding for target classes 
train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)

num_classes = test_y.shape[1]

#creating the seauential model and adding the layers 
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),padding="same",activation="relu",kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation="relu",padding="same",kernel_constraint=MaxNorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation="relu",kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation="softmax"))

#configuring the optimizer and compiling the model 
sgd= SGD(lr=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#viewing model summary 
#model.summary()

#training the model 
model.fit(train_x,train_y,validation_data = (test_x,test_y),epochs=10,batch_size=32)
