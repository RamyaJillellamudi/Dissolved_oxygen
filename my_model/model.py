import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import pickle 

df = pd.read_csv('Dataset1.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Assuming you want to use 8 time steps, you can modify this accordingly
num_timesteps = 8

# Reshape X to have 3 dimensions
X = np.reshape(X, (X.shape[0], num_timesteps, -1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
model.add(GRU(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history=model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
_, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

confusion_mat = confusion_matrix(y_test, y_pred_binary)
print('Confusion Matrix:')
print(confusion_mat)
classification_rep = classification_report(y_test, y_pred_binary)
print('Classification Report:')
print(classification_rep)
print(model.summary())
y_pred_binary = (y_pred > 0.5).astype(int)

# Iterate over the predicted labels
# for i in range(len(y_pred_binary)):
#     # If the label is 0, issue an alert
#     if y_pred_binary[i] == 0:
#         print(f"Alert: Dissolved oxygen is contaminated at index {i}")
#     else:
#         print(f"No alert at index {i}")
# def predict_alert(model, input_parameters):
#     # Predict the output
#     y_pred = model.predict(input_parameters)
    
#     # Convert the prediction to binary
#     y_pred_binary = (y_pred > 0.5).astype(int)
    
#     # Initialize an empty list to store the alert messages
#     alert_messages = []
    
#     # Iterate over the predicted labels
#     for i in range(len(y_pred_binary)):
#         # If the label is 0, issue an alert
#         if y_pred_binary[i] == 0:
#             alert_messages.append(f"Alert: Dissolved oxygen is contaminated at index {i}")
#         else:
#             alert_messages.append(f"No alert at index {i}")
    
#     # Return the alert messages
#     return alert_messages
# alert_messages = predict_alert(model, input_parameters)
# for message in alert_messages:
#     print(message)


rolling_window_size = 10
train_loss_smooth = np.convolve(history.history['loss'], np.ones(rolling_window_size)/rolling_window_size, mode='valid')
val_loss_smooth = np.convolve(history.history['val_loss'], np.ones(rolling_window_size)/rolling_window_size, mode='valid')

# Plot the smoothed curves with color and line type customization
from matplotlib import pyplot as plt
plt.plot(train_loss_smooth, 'r--')
plt.plot(val_loss_smooth, 'b-')
#plt.title('Model Accuracy')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.show()
from scipy.signal import savgol_filter

train_acc_smooth = savgol_filter(history.history['accuracy'], 15, 3)
val_acc_smooth = savgol_filter(history.history['val_accuracy'], 15, 3)

plt.plot(train_acc_smooth, 'r--')
plt.plot(val_acc_smooth, 'b-')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train_accuracy', 'val_accuracy'], loc='lower right')
plt.show()
import matplotlib.pyplot as plt

# Train the model
#history = model.fit(features, labels, epochs=100, batch_size=32, verbose=0, validation_split=0.2)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Smooth the training and validation loss curves
train_loss_smooth = savgol_filter(history.history['loss'], 51, 3)
val_loss_smooth = savgol_filter(history.history['val_loss'], 51, 3)

# Plot the smoothed curves
plt.plot(train_loss_smooth, color='green')
plt.plot(val_loss_smooth, color='blue')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
rolling_window_size = 10
train_loss_smooth = np.convolve(history.history['loss'], np.ones(rolling_window_size)/rolling_window_size, mode='valid')
val_loss_smooth = np.convolve(history.history['val_loss'], np.ones(rolling_window_size)/rolling_window_size, mode='valid')

# Plot the smoothed curves
plt.plot(train_loss_smooth)
plt.plot(val_loss_smooth)
#plt.title('Model Accuracy')
plt.plot(train_loss_smooth, 'g-')
plt.plot(val_loss_smooth, 'b.')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.show()


# Save the model in TensorFlow SavedModel format (pb format)
# model.save('C:\water proj\my_model')
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
