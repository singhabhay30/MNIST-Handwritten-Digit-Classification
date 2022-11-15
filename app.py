from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from numpy import argmax

# Replace Filename with the image file name

img = load_img(filename, grayscale=True, target_size=(28, 28))
img = img_to_array(img)
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')
img = img/255.0

model = load_model('model.h5')
predict_value = model.predict(img)
digit = argmax(predict_value)
print(digit)
