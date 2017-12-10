import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from time import time
from keras.callbacks import TensorBoard


model = Sequential()

#keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


model.add(Dense(units=64, activation='relu', input_dim=4))
model.add(Dense(units=3, activation='softmax'))

#   model.compile(loss='sparse_categorical_crossentropy',
#                 optimizer='sgd',
#                 metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

X,y = load_iris(return_X_y=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model.fit(X_train, y_train, epochs=125, batch_size=32, callbacks=[tensorboard])

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)

classes = model.predict(X_test, batch_size=128)

