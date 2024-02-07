import json
import numpy as np
from keras.models import load_model

# Cargar el modelo entrenado
model = load_model('trained_model.h5')

# Leer los datos preprocesados de prueba desde el archivo JSON
with open('Datos/preprocessed_test_data.json', 'r') as file:
    test_data = np.array(json.load(file))

# Convertir los datos de prueba en un arreglo numpy
X_test = np.array(test_data)

# Agregar una dimensión adicional a los datos de prueba
X_test = np.expand_dims(X_test, axis=1)

# Realizar predicciones en los datos de prueba
predictions = model.predict(X_test)

# Obtener el índice de la neurona con el valor más alto en cada predicción
output = np.argmax(predictions, axis=1)


# Guardar las predicciones en un archivo JSON
with open('Datos/predictions.json', 'w') as file:
    json.dump({"target": output.tolist()}, file)

print("Predicciones guardadas en el archivo 'predictions.json'.")