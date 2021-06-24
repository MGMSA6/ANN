# Artificial Neural Network(ANN)

# Part 1 - Data Preprocessing

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, ReLU
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
pd.set_option('display.max_columns', None)

# Importing the dataset
dataset = pd.read_csv('Dataset/Churn_Modelling.csv')

print(dataset.head(10))

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Create dummy variables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Concatenate the Data Frames
X = pd.concat([X, geography, gender], axis=1)

# Drop Unnecessary columns
X = X.drop(['Geography', 'Gender'], axis=1)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Define model
model = Sequential()

# Adding the first input layer
model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_shape=(11,)))

# Adding the hidden layer
model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

# Adding the last/output layer
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

# Compiling the ANN
model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
model_history = model.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)

# list all data in history
print(model_history.history.keys())

# plot accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Calculate the Accuracy
score = accuracy_score(y_pred, y_test)
print(score)

