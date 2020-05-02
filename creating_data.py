import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os 
import random
import pickle
 
DataDir = 'C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\PetImages'
categories = ['Dog', 'Cat'] 

for category in categories:
    path = os.path.join(DataDir, category) #going in the folder of images, and letting them join the DataDir
    for img in os.listdir(path): #iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE) #reading the image in the form of grayscale and converting it into an array
        plt.imshow(img_array, cmap='gray') #creates an image from an array
        plt.show()
        break
    break 

img_size = 100

new_array = cv2.resize(img_array, (img_size, img_size)) #resizing all the images to a same size i.e., (100 x 100)
plt.imshow(new_array, cmap = 'gray')
plt.show()

training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(DataDir, category)
        class_num = categories.index(category) # 0 for dog and 1 for cat

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size,img_size)) 
                training_data.append([new_array, class_num])

            except Exception as e:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)

X = []
Y = []

for features, labels in training_data:
    X.append(features)
    Y.append(labels)

print(X[0].reshape(-1, img_size, img_size, 1))
X = np.array(X).reshape(-1, img_size, img_size, 1)
Y = np.array(Y).reshape(-1, 1)

pickle_out = open('C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\X1.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\Y1.pickle', 'wb')
pickle.dump(Y, pickle_out)
pickle_out.close()

