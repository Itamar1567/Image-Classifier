import numpy as np
from tensorflow.keras import models, layers, datasets
import cv2 as cv
import matplotlib.pyplot as plt

(trainingImages, trainingLabels), (testingImages, testingLabels) = datasets.cifar10.load_data()
trainingImages, testingImages = trainingImages / 255, testingImages / 255

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#Run through 16 images
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(trainingImages[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[trainingLabels[i][0]])

#Limit how many images are given to the nueral network for speed
trainingImages = trainingImages[:20000]
trainingLabels = trainingLabels[:20000]
testingImages = testingImages[:4000]
testingLabels = trainingLabels[:4000]

model = models.load_model('image_classifier.keras')

img = cv.imread('ImagesForImageClassifier/32Car.jpg')
img = cv.resize(img, (32,32))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')

plt.show()

#                               Training Equation
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #Epochs how many times the model is going to see the same data
# model.fit(trainingImages, trainingLabels, epochs=10, validation_data=(testingImages, testingLabels))
#
# loss, accuracy = model.evaluate(testingImages, testingLabels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")
#
# model.save('image_classifier.keras')




