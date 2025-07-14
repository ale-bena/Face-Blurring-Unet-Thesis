import tensorflow as tf
import numpy as np
from PIL import Image
import os

model_path = "/content/drive/MyDrive/face_blur_project-main/models_mix/unet_mix2.keras"
images_folder = "/content/drive/MyDrive/face_blur_project-main/Mix_data/validation" # non blurred images
output_tflite = "/content/blurring_model_int8.tflite"
input_size = (128, 128)
quant_dtype = tf.uint8 # tf.int8 o tf.uint8
max_images = 200

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

model = tf.keras.models.load_model(model_path,
        custom_objects={
            "ssim_metric": ssim_metric,
            "psnr_metric": psnr_metric
        })

def representation():
    images = sorted(os.listdir(images_folder))
    for fole_name in images[:max_images]:
        try:
            path = os.path.join(images_folder, file_name)
            img = Image.open(percorso).convert("RGB")
            img = img.resize(input_size)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            yield [img_array]
        except Exception as e:
            print(f"Error image {file_name}: {e}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converte.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representation
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = quant_dtype
converter.inference_output_type = quant_dtype

tflite_model = converter.convert()

with open(output_tflite, "wb") as f:
    f.write(tflite_model)

print(f"Conversion completed: saved in {output_tflite}")
