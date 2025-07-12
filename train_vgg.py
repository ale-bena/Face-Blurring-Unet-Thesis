import os
import argparse
import tensorflow as tf
from model import build_blur_unet

# Parametri
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
#EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

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


def load_dataset(image_dir, target_dir, augment_data=False, max_images=None):
    # image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    # target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
    image_files = tf.data.Dataset.list_files(os.path.join(image_dir, "*.jpg"), shuffle=False)
    target_files = tf.data.Dataset.list_files(os.path.join(target_dir, "*.jpg"), shuffle=False)

    if max_images is not None:
        image_files = image_files.take(max_images)
        target_files = target_files.take(max_images)

    # dataset = tf.data.Dataset.from_tensor_slices((image_files, target_files))
    # dataset = dataset.map(process_path, num_parallel_calls=4)
    dataset = tf.data.Dataset.zip((image_files, target_files))
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)

    #if augment_data:
    #    dataset = dataset.map(augment, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    #cdataset = dataset.prefetch(buffer_size=4)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def train_model(resume_training, model_path, epochs):
    if resume_training:
        print(f"Loading existing model from {model_path}...")
        # model = tf.keras.models.load_model(model_path)
        model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ssim_metric": ssim_metric,
            "psnr_metric": psnr_metric
        })
    else:
        print("Creating new model...")
        model = build_blur_unet()

    model.compile(
        optimizer='adam',
        loss='mae',  #probably changing it for a combined one
        metrics=['mae', ssim_metric, psnr_metric]
    )


    train_dataset = load_dataset("./vggface/train", "./vggface/train_blur", max_images=5000)
    validation_dataset = load_dataset("./vggface/val", "./vggface/val_blur", max_images=1000)
    steps_per_epoch = 5000 // BATCH_SIZE
    validation_steps = 1000 // BATCH_SIZE
    #checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    #    "best_model.keras", save_best_only=True, monitor="val_mae", mode="min"
    #)
    #earlystop_cb = tf.keras.callbacks.EarlyStopping(
    #    monitor="val_mae", patience=20, restore_best_weights=True, mode="min"
    #)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "models_vgg/best_model1.keras", save_best_only=True, monitor="val_ssim_metric", mode="max"
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
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
    )
    print("📉 Training completed.")
    
    # Save
    model.save("models_vgg/unet1.keras")
    print("✅ Model saved")


def main(resume_training, model_path, epochs):
    print(f"Starting training with {epochs} epochs")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory GPU growth enabled")
        except RuntimeError as e:
            print(e)

    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    for d in devices:
        if d.device_type == 'GPU':
            print(f"GPU: {d.name}, memory limit: {d.memory_limit / (1024 ** 3):.2f} GB")
    
    train_model(resume_training, model_path, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--resume_training", type=bool, default=False, help="Resume training")
    parser.add_argument("--model_path", type=str, default="models_mix/unet_mix.keras", help="Model path")
    parser.add_argument("--epochs", type=int, help="Num epochs")

    args = parser.parse_args()
    main(args.resume_training, args.model_path, args.epochs)
