import cv2
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

categories = ['DOG', 'CAT']
image = 'C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\Test_Images\\2.jpg'

def prepare(filepath):
    img_size = 100
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)

model = tensorflow.keras.models.load_model('C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\Dogs_vs_Cats_CNN.h5')
result = prepare(image)
#print(result.dtype)
result2 = result.astype('float64')
#print(result2.dtype)

prediction = model.predict([result2])
print(categories[int(prediction[0][0])])
img = mpimg.imread(image)
imgplot = plt.imshow(img)
plt.title(categories[int(prediction[0][0])])
plt.show()