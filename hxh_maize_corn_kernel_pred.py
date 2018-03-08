import numpy as np
import cv2
import os
from keras.models import model_from_json
test_imgs = np.load('dataset/test_data.npy')
model = model_from_json(open('result/cnn_basic.json').read())
model.load_weights('result/cnn.hdf5')
pred = model.predict_classes(test_imgs, verbose=2)
np.save('result/pred.npy', pred)
