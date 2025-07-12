import os
import argparse
import tensorflow as tf
from model import build_blur_unet

# Parametri
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
#EPOCHS = 20

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def process_path(img_path, tgt_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    tgt = tf.io.read_file(tgt_path)
    tgt = tf.image.decode_png(tgt, channels=3)
    tgt = tf.image.resize(tgt, IMG_SIZE)
    tgt = tf.cast(tgt, tf.float32) / 255.0

    return img, tgt


def augment(img, tgt):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    # img = tf.image.random_jpeg_quality(img, 75, 100)
    return img, tgt


def load_dataset(image_dir, target_dir, augment_data=False):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_files, target_files))
    dataset = dataset.map(process_path, num_parallel_calls=4)

    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=4)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=4)
    return dataset

def train_model(resume_training, model_path, epochs):
    if resume_training:
        print(f"Caricamento modello esistente da {model_path}...")
        # model = tf.keras.models.load_model(model_path)
        model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ssim_metric": ssim_metric,
            "psnr_metric": psnr_metric
        })
    else:
        print("Creazione modello nuovo...")
        model = build_blur_unet()

    model.compile(
        optimizer='adam',
        loss='mae',  # oppure 'mse'
        metrics=['mae', ssim_metric, psnr_metric]
    )


    train_dataset = load_dataset("./Mix_data/train", "./Mix_data/train_blurred", augment_data=True)
    validation_dataset = load_dataset("./Mix_data/validation", "./Mix_data/validation_blurred")

    #checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    #    "best_model.keras", save_best_only=True, monitor="val_mae", mode="min"
    #)
    #earlystop_cb = tf.keras.callbacks.EarlyStopping(
    #    monitor="val_mae", patience=20, restore_best_weights=True, mode="min"
    #)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "models_mix/best_model2.keras", save_best_only=True, monitor="val_ssim_metric", mode="max"
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_ssim_metric", patience=15, restore_best_weights=True, mode="max"
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, verbose=1
    )

    # Training
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
    )
    print("📉 Training completato.")
    
    # Save
    model.save("models_mix/unet_mix3.keras")
    print("✅ Modello salvato come prova")


def main(resume_training, model_path, epochs):
    print(f"👉 Inizio training con {epochs} epoche")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memoria GPU growth abilitata")
        except RuntimeError as e:
            print(e)

    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    for d in devices:
        if d.device_type == 'GPU':
            print(f"🧠 GPU trovata: {d.name}, Limite memoria: {d.memory_limit / (1024 ** 3):.2f} GB")
    
    train_model(resume_training, model_path, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--resume_training", type=bool, default=False, help="Resume training")
    parser.add_argument("--model_path", type=str, default="models_mix/unet_mix.keras", help="Model path")
    parser.add_argument("--epochs", type=int, help="Num epochs")

    args = parser.parse_args()
    main(args.resume_training, args.model_path, args.epochs)
