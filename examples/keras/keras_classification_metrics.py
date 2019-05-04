# from numpy import array
# from keras.models import Sequential
# from keras.layers import Dense
# from matplotlib import pyplot
# # prepare sequence
# X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# y = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# # create model
# model = Sequential()
# model.add(Dense(2, input_dim=1))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# # train model
# history = model.fit(X, y, epochs=400, batch_size=len(X), verbose=2)
# # plot metrics
# pyplot.plot(history.history['acc'])
# pyplot.show()
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras.optimizers import SGD
# prepare sequence
X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# create model
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.00001, momentum=1.0), metrics=['acc'])
# train model
history = model.fit(X, y, epochs=400, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['acc'])
pyplot.show()


from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Add a binary classification layer (sigmoid)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)



