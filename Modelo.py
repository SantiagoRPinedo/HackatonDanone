import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt

# Leer los datos preprocesados desde el archivo JSON
with open('Datos/preprocessed_train_data.json', 'r') as file:
    data = json.load(file)

# Convertir los datos en un arreglo numpy
X_train = np.array(data)

# Agregar una dimensión adicional a los datos de entrenamiento
X_train = np.expand_dims(X_train, axis=1)

# Obtener el número de timesteps (longitud de la secuencia) a partir de los datos
timesteps = X_train.shape[1]

# Leer las etiquetas de entrenamiento desde el archivo JSON
with open('Datos/etiquetas_entrenamiento.json', 'r') as file:
    labels = json.load(file)

# Convertir las etiquetas en un arreglo numpy resultados esperados
y_train = np.array(labels)

# Reshape los datos de entrada para que tengan la forma adecuada para Conv2D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

# Convertir las etiquetas en formato one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)


# Definir el modelo
model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu", input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(Flatten())
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(5, activation="softmax"))


# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=64)

# Obtener las métricas de entrenamiento
accuracy = history.history['accuracy']
loss = history.history['loss']

# Crear la gráfica
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'b', label='Precisión de entrenamiento')
plt.plot(epochs, loss, 'r', label='Pérdida de entrenamiento')
plt.title('Precisión y Pérdida de entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión / Pérdida')
plt.legend()
plt.show()

# Guardar el modelo entrenado
model.save('trained_model.h5')