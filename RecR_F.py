# ----------------------------------------------------------------------
# PASO 0: IMPORTAR LIBRERÍAS
# ----------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # <--- Añade esta línea
import matplotlib.pyplot as plt
import datetime

# Importar para KPIs (Métricas)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Importar para Probar (PIL y io)
from PIL import Image, ImageOps
import io

print(f"TensorFlow Version: {tf.__version__}")
print("Librerías importadas.")

# ----------------------------------------------------------------------
# PASO 1: DEFINIR RUTAS Y CONSTANTES
# ----------------------------------------------------------------------
print("\n--- Configurando Rutas ---")

# La ruta ahora es local, relativa a donde ejecutas el script.
BASE_DATA_DIR = "./Reconocimiento_Facial"
TEST_IMG_DIR = "./TEST_IMAGES" # <--- ¡NUEVA LÍNEA!

# Guardaremos el modelo DENTRO de la carpeta del proyecto
MODEL_SAVE_PATH = os.path.join(BASE_DATA_DIR, "reconocimiento_facial_model.keras")
LOG_DIR = "./facial_recognition_logs" # Logs de TensorBoard

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True) # <--- ¡NUEVA LÍNEA!

# Comprobar si la carpeta de datos existe
if not os.path.exists(BASE_DATA_DIR):
    raise FileNotFoundError(
        f"Error: No se pudo encontrar la carpeta del dataset en {BASE_DATA_DIR}\n"
        "Por favor, asegúrate de haber descargado y descomprimido la carpeta 'Reconocimiento_Facial' "
        "en el mismo directorio que este script."
    )

print(f"Ruta del Dataset: {BASE_DATA_DIR}")
print(f"El modelo se guardará en: {MODEL_SAVE_PATH}")
print(f"Carpeta de prueba lista en: {TEST_IMG_DIR}")

# ----------------------------------------------------------------------
# PASO 2: CARGAR Y PREPARAR DATOS (OPTIMIZADO PARA 8GB RAM)
# ----------------------------------------------------------------------
# --- ¡CAMBIOS CLAVE PARA RAM! ---
IMG_SIZE = (128, 128) # Reducido para 8GB RAM
BATCH_SIZE = 32       # Lotes más pequeños
EPOCHS = 10           # 10 épocas es un buen inicio
VALIDATION_SPLIT = 0.2 # 20% de los datos para validación

print(f"\nOptimizando para 8GB RAM: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}")

# 1. Cargar el dataset de ENTRENAMIENTO (80% de los datos)
train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DATA_DIR,
    label_mode='int',
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 2. Cargar el dataset de VALIDACIÓN (20% de los datos)
val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DATA_DIR,
    label_mode='int',
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"¡Éxito! Encontradas {NUM_CLASSES} clases (integrantes): {class_names}")

# 3. Optimizar el Pipeline de Datos (¡SIN .cache()!)
AUTOTUNE = tf.data.AUTOTUNE
# Eliminamos .cache() para respetar los 8GB de RAM.
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Pipelines de datos optimizados con .prefetch() (¡.cache() eliminado por RAM!)")


# ----------------------------------------------------------------------
# PASO 3: CONFIGURAR TENSORBOARD
# ----------------------------------------------------------------------
print("\n--- Configurando TensorBoard ---")

log_dir_run = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_run, histogram_freq=1)

print(f"Logs para esta sesión se guardarán en: {log_dir_run}")
# --- ¡NOTA! ---
print(f"Para ver TensorBoard, abre OTRA terminal y ejecuta:")
print(f"tensorboard --logdir={LOG_DIR}")


# ----------------------------------------------------------------------
# PASO 4, 5 y 6: CARGAR O ENTRENAR EL MODELO
# ----------------------------------------------------------------------

# Comprobar si el modelo ya existe en el disco
if os.path.exists(MODEL_SAVE_PATH):
    # --- 1. SI EXISTE: CARGAR EL MODELO ---
    print(f"\n--- Cargando modelo existente desde: {MODEL_SAVE_PATH} ---")
    model = keras.models.load_model(MODEL_SAVE_PATH)
    history = None # No hay historial de entrenamiento si solo cargamos
    print("--- Modelo Cargado ---")
    model.summary()

else:
    # --- 2. SI NO EXISTE: DEFINIR, ENTRENAR Y GUARDAR ---
    print("\n--- No se encontró modelo. Entrenando uno nuevo... ---")
    
    # --- (PASO 4: DEFINIR ARQUITECTURA) ---
    print("\nCreando modelo de CNN con Transfer Learning (MobileNetV2)...")
    
    # Capa de Aumento de Datos
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ],
        name="data_augmentation"
    )

    # Cargar el Modelo Base
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Congelar

    # Construir el Modelo Final
    inputs = keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x) 
    model = keras.Model(inputs, outputs)

    # Compilar el modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # --- (PASO 5: ENTRENAR EL MODELO) ---
    print(f"\n--- Iniciando Entrenamiento ({EPOCHS} épocas) ---")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[tensorboard_callback] 
    )
    print("--- Entrenamiento Completado ---")

    # --- (PASO 6: GUARDAR EL MODELO ENTRENADO) ---
    print(f"\n--- Guardando modelo en disco local ---")
    try:
        model.save(MODEL_SAVE_PATH)
        print(f"¡Modelo guardado exitosamente en: {MODEL_SAVE_PATH}!")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

# ----------------------------------------------------------------------
# PASO 7: EVALUAR Y VISUALIZAR KPIs (CORREGIDO PARA GUARDAR ARCHIVOS)
# ----------------------------------------------------------------------
print("\n--- Evaluando KPIs del Modelo ---")

# 1. Extraer etiquetas reales (y_true) y predicciones (y_pred)
print("Generando predicciones en el set de validación...")
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# 2. Reporte de Clasificación (KPIs)
print("\n--- Reporte de Clasificación (KPIs) ---")
print(classification_report(y_true, y_pred, target_names=class_names, labels=range(NUM_CLASSES), zero_division=0))

# 3. Matriz de Confusión
print("\n--- Matriz de Confusión ---")
cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

fig_size = max(6, NUM_CLASSES * 0.8) 
plt.figure(figsize=(fig_size, fig_size)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# --- ¡CAMBIO CLAVE! ---
# No mostrar, guardar
cm_path = "confusion_matrix.png"
plt.savefig(cm_path)
plt.close() # Cierra la figura de la memoria
print(f"¡Matriz de Confusión guardada en: {cm_path}!")

# 4. Gráficos de Precisión y Pérdida (de Matplotlib)
if history is not None:
    print("\n--- Curvas de Entrenamiento (Matplotlib) ---")
    plt.figure(figsize=(14, 5))

    # Gráfico de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión (train)')
    plt.plot(history.history['val_accuracy'], label='Precisión (val)')
    plt.title('Precisión del Modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend()

    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida (train)')
    plt.plot(history.history['val_loss'], label='Pérdida (val)')
    plt.title('Pérdida del Modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend()

    # --- ¡CAMBIO CLAVE! ---
    # No mostrar, guardar
    curves_path = "training_curves.png"
    plt.savefig(curves_path)
    plt.close() # Cierra la figura de la memoria
    print(f"¡Curvas de Entrenamiento guardadas en: {curves_path}!")
else:
    print("\n--- Saltando gráficas de Matplotlib (Modelo cargado desde disco) ---")

# 5. Evaluación final
loss, accuracy = model.evaluate(val_ds)
print(f"\nPrecisión final de validación: {accuracy * 100:.2f}%")
print("\n--- FIN DE LA EVALUACIÓN ---")

# ----------------------------------------------------------------------
# PASO 8: PRUEBAS DE PREDICCIÓN (DESPLIEGUE - CORREGIDO)
# ----------------------------------------------------------------------

# --- Prueba 1: El modelo muestra una prueba ---
def mostrar_prueba_validacion(val_ds, model, class_names):
    print("\n--- PRUEBA 1: El modelo muestra una predicción ---")
    try:
        images, labels = next(iter(val_ds.unbatch().batch(1))) # Tomar 1 imagen
        img_to_test = images[0]
        label_true_index = labels[0].numpy()
        prediction_probs = model.predict(images, verbose=0)
        prediction_index = np.argmax(prediction_probs[0])
        confidence = np.max(prediction_probs[0]) * 100
        label_true_name = class_names[label_true_index]
        prediction_name = class_names[prediction_index]
        
        plt.figure(figsize=(6,6))
        plt.imshow(((img_to_test + 1) / 2).numpy().astype("uint8"))
        plt.title(f"Real: {label_true_name}\nPredicción: {prediction_name} ({confidence:.2f}%)",
                  color=("green" if label_true_index == prediction_index else "red"))
        plt.axis("off")
        
        demo_path = "demo_prediction.png"
        plt.savefig(demo_path)
        plt.close()
        print(f"¡Predicción de demo guardada en: {demo_path}!")
        
    except Exception as e:
        print(f"Error al mostrar la prueba de validación: {e}")

# --- Prueba 2: Tú subes una imagen ---
def probar_imagen_propia(model, class_names, img_size):
    print("\n--- PRUEBA 2: Prueba con tu propia imagen ---")
    
    # --- ¡CAMBIO CLAVE! ---
    # Ya no pedimos la ruta, solo el nombre del archivo.
    filename = input(f"Introduce el NOMBRE de tu imagen (debe estar en la carpeta 'TEST_IMAGES'): ")
    
    # Construir la ruta completa
    file_path = os.path.join(TEST_IMG_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo '{filename}' en la carpeta '{TEST_IMG_DIR}'.")
        print("Asegúrate de que el nombre esté bien escrito y la imagen esté en esa carpeta.")
        return

    try:
        img = Image.open(file_path)
        img = img.convert('RGB')
        
        # Redimensionar la imagen
        img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)
        
        prediction_probs = model.predict(img_batch, verbose=0)
        prediction_index = np.argmax(prediction_probs[0])
        confidence = np.max(prediction_probs[0]) * 100
        prediction_name = class_names[prediction_index]
        
        plt.figure(figsize=(6,6))
        plt.imshow(img_resized)
        plt.title(f"Tu imagen\nPredicción: {prediction_name} ({confidence:.2f}%)")
        plt.axis("off")
        
        user_pred_path = "user_prediction.png"
        plt.savefig(user_pred_path)
        plt.close()
        print(f"¡Tu predicción guardada en: {user_pred_path}!")
        
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")

# --- Ejecutar las pruebas ---
mostrar_prueba_validacion(val_ds, model, class_names)
probar_imagen_propia(model, class_names, IMG_SIZE)

print("\n--- FIN DEL SCRIPT ---")