#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense


# In[2]:


train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)
epochs = 20
batch_size = 16
nb_train_samples = 4000
nb_validation_samples = 600
nb_test_samples = 600


# In[3]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[4]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[5]:


datagen = ImageDataGenerator(rescale=1. / 255)


# In[6]:


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[7]:


val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[8]:


test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[9]:


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)


# In[10]:


scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)


# In[11]:


print("Accuracy: %.2f%%" % (scores[1]*100))


# In[12]:


import pandas as pd 


# In[13]:


model.to_json()


# In[ ]:




