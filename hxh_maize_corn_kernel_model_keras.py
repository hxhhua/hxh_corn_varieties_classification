from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.layers import Flatten, Activation, Dense, BatchNormalization
from keras.layers.local import LocallyConnected2D

from keras.utils import np_utils
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
import json
from keras import optimizers
n_label = 5

img_w = 64		#	28
img_h = 64		# 28
pool_size1 = (2,2)


def load_dataset():

	train = np.load('dataset/train_data.npy')
	label = np.load('dataset/train_label_data.npy')
	index = [i for i in range(len(train))]
	random.shuffle(index)
	imgs_train = train[index]
	imgs_label = label[index]
	
	X_train= imgs_train.astype('float32')

	encoder = LabelEncoder()
	encoder_y_train = encoder.fit_transform(imgs_label)
	Y_train = np_utils.to_categorical(encoder_y_train, n_label)	
	return X_train, Y_train

def val_load_dataset():
	val = np.load('dataset/val_data.npy')
	val_label = np.load('dataset/val_label_data.npy')
	num = len(val)
	index = [i for i in range(num)]
	random.shuffle(index)
	imgs_val = val[index]
	imgs_label = val_label[index]

	X_val = imgs_val.astype('float32')
	encoder = LabelEncoder()
	encoder_y_val = encoder.fit_transform(imgs_label)
	Y_val = np_utils.to_categorical(encoder_y_val,n_label)
	return X_val, Y_val

def test_load_dataset():
	test = np.load('dataset/test_data.npy')
	test_label = np.load('dataset/test_label_data.npy')
	X_test= test.astype('float32')

	encoder = LabelEncoder()
	encoder_y_test = encoder.fit_transform(test_label)
	Y_test = np_utils.to_categorical(encoder_y_test, n_label)	
	return X_test, Y_test

def VGG():
	model = Sequential()
	# 1
	model.add(Convolution2D(64,3,3,input_shape=(img_w, img_h,3), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	#2
	model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	# 3
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	# 4
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	# 5
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))


	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))


	model.add(Dense(n_label))
	model.add(Activation('softmax'))
	a = optimizers.Adadelta(lr=0.1,decay=0.0005)

	model.compile(loss='categorical_crossentropy',
					optimizer=a,				# 'adadelta',
					metrics=['accuracy'])
	print(model.summary())
	return model

def VGG_BN():
	model = Sequential()
	# 1
	model.add(Convolution2D(64,3,3,input_shape=(img_w, img_h,3), border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(64,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	#2
	model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(128,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	# 3
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(256,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	# 4
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))
	# 5
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(512,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1, strides=(2,2)))


	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	


	model.add(Dense(n_label))
	model.add(Activation('softmax'))
	#sgd = optimizers.SGD(lr =0.01,decay=0.0005, momentum=0.9)
	a = optimizers.Adadelta(lr=0.1,decay=0.0005)

	model.compile(loss='categorical_crossentropy',
					optimizer=a,				# 'adadelta',
					metrics=['accuracy'])

	print(model.summary())
	return model

def LeNet_5():
	model = Sequential()
	model.add(Convolution2D(6,5,5,
							border_mode='valid',
							input_shape=(img_w, img_h,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Convolution2D(16, 5,5))

	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	
	model.add(Flatten())
	
	model.add(Dense(120))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(84))
	model.add(Activation('relu'))
	# softmax classification, the output is 2 categories
	model.add(Dense(n_label))
	model.add(Activation('softmax'))
	sgd = optimizers.SGD(lr =0.01,decay=0.0005, momentum=0.9)

	model.compile(loss='categorical_crossentropy',
					optimizer='adadelta',
					metrics=['accuracy'])
	print(model.summary())
	return model

def AlexNet():
	model = Sequential()
	model.add(Convolution2D(96,11,11, input_shape=(img_h, img_w,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1))

	model.add(Convolution2D(256,5,5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1))

	model.add(Convolution2D(384,3,3))
	model.add(Activation('relu'))
	model.add(Convolution2D(384,3,3))
	model.add(Activation('relu'))
	model.add(Convolution2D(256,3,3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size1))

	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dense(2))
	model.add(Activation('softmax'))



if __name__=='__main__':
	train, label = load_dataset()
	#val, val_label = val_load_dataset()

	#model = LeNet_5()
	model = VGG()
	#model = VGG_BN()
	model_checkpoint = ModelCheckpoint('result/vgg.hdf5',
									monitor = 'val_loss',
									save_best_only= True)
	hist = model.fit(train, label, batch_size=32,
			nb_epoch= 20, verbose= 1, shuffle=False, validation_split=0.2,
			callbacks=[model_checkpoint])	#validation_data=(val, val_label),
	model.save_weights('result/vgg_weight.hdf5')
	with open('result/vgg_basic.json', 'w') as f:
		f.write(json.dumps(json.loads(model.to_json()), indent=2))

	with open('result/vgg_acc.txt','w') as f:
		f.write(str(hist.history))
	print('sucess')
	# predict
	test, test_label = test_load_dataset()
	score = model.evaluate(test, test_label, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	pred = model.predict_classes(test, verbose=2)
	np.save('result/vgg_pred.npy', pred)
	