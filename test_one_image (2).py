import tensorflow as tf
import numpy as np
from PIL import Image

# Funzione per caricare e preprocessare l'immagine
def load_and_preprocess_image(image_path, img_size=(128,128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

# Funzione per eseguire inferenza con modello TFLite
def run_tflite_model(tflite_model_path, input_data):
    # Carica modello TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Ottieni dettagli input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Imposta input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Esegui inferenza
    interpreter.invoke()

    # Prendi output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if __name__ == "__main__":
    tflite_model_path = "/content/drive/MyDrive/BlurProject/face_blur_project/models/blur_unet.keras"
    test_image_path = "/content/drive/MyDrive/BlurProject/face_blur_project/test_img/face.jpg"

    input_data = load_and_preprocess_image(test_image_path)
    output = run_tflite_model(tflite_model_path, input_data)

    # Rimuovi batch e scala output (se serve)
    output_img = np.squeeze(output)
    output_img = (output_img * 255).astype(np.uint8)

    # Salva output come immagine
    Image.fromarray(output_img).save("output_blurred.jpg")
    print("Inferenza completata, risultato salvato in output_blurred.png")
