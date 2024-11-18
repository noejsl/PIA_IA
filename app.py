from flask import Flask, request, render_template, url_for, redirect
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import kagglehub

app = Flask(__name__)

# Función para cargar las imágenes y sus etiquetas
def load_images_and_labels(image_folder):
    images = []
    labels = []

    # Verifica el contenido de la carpeta
    print("Contenido de la carpeta:", os.listdir(image_folder))  # Verifica si hay imágenes en la carpeta

    # Recorrer todos los archivos en la subcarpeta 'samples'
    sample_folder = os.path.join(image_folder, 'samples')
    print("Contenido de la subcarpeta 'samples':", os.listdir(sample_folder))  # Verifica si hay imágenes

    # Recorrer todos los archivos en la carpeta 'samples'
    for filename in os.listdir(sample_folder):
        if filename.endswith(".png"):  # Asegúrate de que sea una imagen
            img_path = os.path.join(sample_folder, filename)

            # Cargar la imagen y redimensionarla a 200x50
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(50, 200))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalizar la imagen

            images.append(img_array)

            # Extraer el nombre del archivo como etiqueta (sin la extensión)
            label = filename.split(".")[0]
            labels.append(label)

    return np.array(images), np.array(labels)

# Función para cargar el modelo desde la carpeta 'static'
def load_model():
    model_path = os.path.join(app.static_folder, 'modelo_completo.keras')  # Ruta al archivo en static
    model = tf.keras.models.load_model(model_path)
    return model

# Descargar el conjunto de datos de Kaggle
image_folder = kagglehub.dataset_download("fournierp/captcha-version-2-images")  # Directamente la ruta

# Verifica la ruta descargada
print("Ruta de las imágenes:", image_folder)

# Cargar las imágenes y las etiquetas
images, labels = load_images_and_labels(image_folder)

# Verifica cuántas imágenes fueron cargadas
print("Número de imágenes cargadas:", len(images))

# Preprocesamiento de etiquetas
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)  # Tokenización por carácter
tokenizer.fit_on_texts(labels)  # Ajusta el tokenizador a las etiquetas

num_classes = len(tokenizer.word_index) + 1  # Número total de clases (incluyendo el padding)

# Convertir las etiquetas a secuencias numéricas
y_labels = tokenizer.texts_to_sequences(labels)

# Asegurarse de que todas las secuencias de etiquetas tengan la misma longitud (en este caso 5 caracteres)
y_labels = [seq + [0] * (5 - len(seq)) for seq in y_labels]  # Padding
y_labels = np.array(y_labels)

# Convertir a one-hot encoding
y_labels = np.array([tf.keras.utils.to_categorical(label, num_classes=num_classes) for label in y_labels])

# Cargar el modelo desde el archivo .h5
modelo = load_model()

# Función para cargar la imagen y predecir el texto
def predict_image(file, model, tokenizer):
    # Leer la imagen desde el archivo en memoria
    img_bytes = file.read()  # Lee los bytes del archivo cargado
    img = Image.open(io.BytesIO(img_bytes))  # Abre la imagen desde los bytes usando PIL

    # Convertir la imagen a RGB (en caso de que tenga 4 canales, RGBA, eliminamos el canal alfa)
    img = img.convert('RGB')  # Convertimos a RGB para que tenga solo 3 canales

    # Redimensionar la imagen y convertirla a un array
    img = img.resize((200, 50))  # Asegúrate de que la imagen tiene el tamaño correcto
    img_array = np.array(img) / 255.0  # Normalización

    # Asegurarse de que la imagen tenga la forma (1, 50, 200, 3) para el modelo
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción
    predictions = model.predict(img_array)

    # Decodificar las predicciones (convertir de one-hot encoding a texto)
    predicted_text = ''
    for i in range(5):  # El modelo predice 5 caracteres
        predicted_class = np.argmax(predictions[i], axis=-1)  # Índice de la clase con mayor probabilidad
        # Convertir la clase a su caracter correspondiente
        predicted_char = tokenizer.index_word.get(int(predicted_class), '?')  # Asegurarse de que sea un int
        predicted_text += predicted_char

    # Mostrar la imagen y la predicción
    plt.imshow(img)
    plt.title(f'Predicción: {predicted_text}')
    plt.axis('off')  # No mostrar los ejes
    plt.show()

    return predicted_text

# Ruta para mostrar el formulario
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen desde la solicitud POST
    archivo_imagen = request.files['image']

    # Realizar la predicción con la imagen cargada
    prediccion = predict_image(archivo_imagen, modelo, tokenizer)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
