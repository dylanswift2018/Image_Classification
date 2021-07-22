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

