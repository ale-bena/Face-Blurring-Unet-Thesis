import tensorflow as tf

# Percorso modello SavedModel
saved_model_dir = "saved_model/blur_unet"

# Percorso dove salvare il file tflite
tflite_model_path = "blur_unet.tflite"

# Crea il convertitore TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# (Opzionale) Ottimizzazioni per edge device, es:
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Converte il modello
tflite_model = converter.convert()

# Salva il modello TFLite su file
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Modello TFLite salvato in: {tflite_model_path}")

