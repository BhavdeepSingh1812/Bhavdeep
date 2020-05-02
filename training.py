import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle

pickle_in = open('C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\X1.pickle', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\Y1.pickle', 'rb')
Y = pickle.load(pickle_in)

X = X/255.0  # normalising the values of features between 0 & 1 (Maximum RGB value = 255)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(X.shape, X.dtype)
#print(Y.shape, Y.dtype)

model.fit(X, Y, batch_size = 4, epochs = 10, verbose=1, validation_split=0.3)

model.save('C:\\Users\\Bhavdeep\\Desktop\\Bhavdeep\\Cats_Dogs_Classifier\\Dogs_vs_Cats_CNN.h5')