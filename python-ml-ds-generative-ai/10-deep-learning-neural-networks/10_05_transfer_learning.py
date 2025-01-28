
# coding: utf-8

# # Transfer Learning
# 
# Using pre-trained models in Keras is really easy.
# 
# Let's use the ResNet50 model, trained on the imagenet data set, in order to quickly classify new images.
# 
# Let's start with a picture of a fighter jet I took while exploring the deserts of California:

# In[1]:


from IPython.display import Image
Image(filename='fighterjet.jpg') 


# Let's load up the modules we need...

# In[2]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


# The ResNet50 pre-trained CNN expects inputs of 224x224 resolution, and will classify objects into one of 1,000 possible categories.
# 
# Let's load up our picture of a fighter jet, rescale it to the resolution the model requires, and use the model's preprocess_input function to further normalize the image data before feeding it in as input.

# In[3]:


img_path = 'fighterjet.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# Let's load up the model itself:

# In[4]:


model = ResNet50(weights='imagenet')


# It's already trained with weights learned from the Imagenet data set. So all we have to do now is use it! We can call predict() on it, just like we would with any machine learning model now:

# In[5]:


preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])


# And, it worked! Let's put all of this into a function so we can quickly classify other images:

# In[6]:


def classify(img_path):
    display(Image(filename=img_path))
    
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])


# Let's see if we can stump it. Here are a few other random photos I had in my personal gallery:

# In[7]:


classify('bunny.jpg')


# In[8]:


classify('firetruck.jpg')


# In[9]:


classify('breakfast.jpg')


# In[10]:


classify('castle.jpg')


# In[11]:


classify('VLA.jpg')


# In[12]:


classify('bridge.jpg')


# That's pretty impressive.
# 
# ## Your Challenge
# 
# Try some photos of your own!
# 
# And, try some different pre-trained models. In addition to ResNet50, Keras offers others such as Inception and MobileNet. Refer to the documentation at https://keras.io/applications/ and see if you can get them working as well. Bear in mind different models will have different requirments for the input image size.
