import json
import numpy as np
import pandas as pd

# Función para cargar los datos desde un archivo JSON
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Función para hacer numéricos los materiales
def materiales(data):
    materials_df = pd.DataFrame(columns=['material', 'numeric_value'])
    numeric_values_list = []

    for item in data.values():
        packaging_materials = item.get('packaging_materials', [])
        numeric_values = []

        for material in packaging_materials:
            if material not in materials_df['material'].values:
                numeric_value = len(materials_df) + 1
                materials_df.loc[len(materials_df)] = [material, numeric_value]
            else:
                numeric_value = materials_df.loc[materials_df['material'] == material, 'numeric_value'].values[0]

            numeric_values.append(numeric_value)

        numeric_values_list.append(numeric_values)

    return numeric_values_list

# Función para preprocesar los datos y convertirlos en un formato adecuado para la red neuronal
def preprocess_data(data, mats):
    processed_data = []
    labels = []

    for i, item in enumerate(data.values()):
        packaging_materials = item.get('packaging_materials', [])
        is_beverage = float(item.get('is_beverage', -1)) if item.get('is_beverage', '') != 'unknown' else -1
        selling_countries = item.get('selling_countries', [])
        non_recyclable_and_non_biodegradable_materials_count = float(item.get('non_recyclable_and_non_biodegradable_materials_count', -1)) if item.get('non_recyclable_and_non_biodegradable_materials_count', '') != 'unknown' else -1
        est_co2_agriculture = float(item.get('est_co2_agriculture', -1)) if item.get('est_co2_agriculture', '') != 'unknown' else -1
        est_co2_consumption = float(item.get('est_co2_consumption', -1)) if item.get('est_co2_consumption', '') != 'unknown' else -1
        est_co2_distribution = float(item.get('est_co2_distribution', -1)) if item.get('est_co2_distribution', '') != 'unknown' else -1
        est_co2_packaging = float(item.get('est_co2_packaging', -1)) if item.get('est_co2_packaging', '') != 'unknown' else -1
        est_co2_processing = float(item.get('est_co2_processing', -1)) if item.get('est_co2_processing', '') != 'unknown' else -1
        est_co2_transportation = float(item.get('est_co2_transportation', -1)) if item.get('est_co2_transportation', '') != 'unknown' else -1

        label = item.get('ecoscore_grade', -1)
        labels.append(label)

        feature_vector = [
            len(selling_countries), is_beverage, len(packaging_materials),
            non_recyclable_and_non_biodegradable_materials_count,
            est_co2_agriculture, est_co2_consumption, est_co2_distribution,
            est_co2_packaging, est_co2_processing, est_co2_transportation
        ]

        mats_vector = mats[i]  # Obtener el vector de materiales correspondiente al producto

        combined_vector = feature_vector + mats_vector
        processed_data.append(combined_vector)

    max_length = max(len(vector) for vector in processed_data)

    padded_data = []
    for vector in processed_data:
        padding_length = max_length - len(vector)
        padded_vector = vector + [0] * padding_length
        padded_data.append(padded_vector)

    return np.array(padded_data), np.array(labels)

# Ruta de los archivos de datos
train_data_file = 'Datos/train_products.json'
test_data_file = 'Datos/test_products.json'

# Cargar los datos de entrenamiento y prueba
train_data = load_data(train_data_file)
test_data = load_data(test_data_file)

# Extraer materiales
mats = materiales(train_data)

# Preprocesar los datos de entrenamiento y prueba
train_features, train_labels = preprocess_data(train_data, mats)
test_features, test_labels = preprocess_data(test_data, mats)

# Crear la columna adicional
extra_column = np.full((test_features.shape[0], 1), 0)

# Concatenar la columna adicional a los vectores de características
test_features = np.concatenate((test_features, extra_column), axis=1)

# Convertir a numpy array
test_features = np.array(test_features)

# Rutas de los archivos de salida
train_output_file = 'Datos/preprocessed_train_data.json'
test_output_file = 'Datos/preprocessed_test_data.json'
labels_output_file = 'Datos/etiquetas_entrenamiento.json'
materiales_numericos = 'Datos/materiales_numericos.json'

# Guardar los datos preprocesados de entrenamiento en un archivo JSON
with open(train_output_file, 'w') as f:
    json.dump(train_features.tolist(), f)

# Guardar los datos preprocesados de salida entrenamiento en un archivo JSON
with open(labels_output_file, 'w') as f:
    json.dump(train_labels.tolist(), f)

# Guardar los datos preprocesados de prueba en un archivo JSON
with open(test_output_file, 'w') as f:
    json.dump(test_features.tolist(),f)


# Imprimir las dimensiones de los conjuntos de datos
print("Dimensiones de los conjuntos de datos:")
print("Datos de entrenamiento:", train_features.shape)
print("Salidas del entrenamiento:", train_labels.shape)
print("Datos de prueba:", test_features.shape)
