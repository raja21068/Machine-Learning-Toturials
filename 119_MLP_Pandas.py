import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


# from keras.utils.visualize_util import plot
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot


# load pima indians dataset
dataset = pd.read_csv('Chapter6/Data/Diabetes.csv')

# split into input (X) and output (y) variables
X = dataset.ix[:,0:8].values
y = dataset['class'].values     # dependent variables

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))



# Fit the model
model.fit(X, y, nb_epoch=5, batch_size=10)
# evaluate the model
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))