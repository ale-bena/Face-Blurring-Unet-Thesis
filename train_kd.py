import os
import argparse
import tensorflow as tf
from model import build_blur_unet as build_teacher
from model_lite import build_blur_unet as build_student

IMG_SIZE = (128, 128)
BATCH_SIZE = 4
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

def load_dataset(image_dir, target_dir, max_images=None):
    image_files = tf.data.Dataset.list_files(os.path.join(image_dir, "*.jpg"), shuffle=False)
    target_files = tf.data.Dataset.list_files(os.path.join(target_dir, "*.jpg"), shuffle=False)

    if max_images is not None:
        image_files = image_files.take(max_images)
        target_files = target_files.take(max_images)

    dataset = tf.data.Dataset.zip((image_files, target_files))
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def distillation_loss(y_true, y_pred_student, y_pred_teacher, alpha=0.5):
    mae_gt = tf.reduce_mean(tf.abs(y_true - y_pred_student))
    mse_teacher = tf.reduce_mean(tf.square(y_pred_teacher - y_pred_student))
    return alpha * mae_gt + (1 - alpha) * mse_teacher

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, alpha=0.5):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha

    def compile(self, optimizer, metrics):
        super(Distiller, self).compile()
        self.optimizer = optimizer
        self.compiled_metrics = metrics

    def train_step(self, data):
        x, y_true = data
        teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            loss = distillation_loss(y_true, student_pred, teacher_pred, self.alpha)

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.compiled_metrics.update_state(y_true, student_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

    def test_step(self, data):
        x, y_true = data
        student_pred = self.student(x, training=False)
        self.compiled_metrics.update_state(y_true, student_pred)
        return {m.name: m.result() for m in self.metrics}

def train_model(teacher_model_path, epochs):
    print("Loading pretrained teacher model...")
    teacher = tf.keras.models.load_model(
        teacher_model_path,
        custom_objects={"ssim_metric": ssim_metric, "psnr_metric": psnr_metric}
    )
    teacher.trainable = False

    print("Building student model...")
    student = build_student()

    distiller = Distiller(student=student, teacher=teacher, alpha=0.5)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            ssim_metric,
            psnr_metric
        ]
    )

    print("Loading datasets...")
    train_dataset = load_dataset("./vggface/train", "./vggface/train_blur", max_images=5000)
    val_dataset = load_dataset("./vggface/val", "./vggface/val_blur", max_images=1000)

    steps_per_epoch = 5000 // BATCH_SIZE
    validation_steps = 1000 // BATCH_SIZE

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models_vgg/best_student.keras", save_best_only=True,
            monitor="val_ssim_metric", mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ssim_metric", patience=15, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, verbose=1
        )
    ]

    print("Starting Knowledge Distillation training...")
    distiller.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    student.save("models_vgg/student_blur_unet.keras")
    print("Student model saved to models_vgg/student_blur_unet.keras")

def main(teacher_model_path, epochs):
    print(f"Training student with teacher at: {teacher_model_path}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(e)

    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    for d in devices:
        if d.device_type == 'GPU':
            print(f"GPU: {d.name}, memory: {d.memory_limit / (1024 ** 3):.2f} GB")

    train_model(teacher_model_path, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training with Knowledge Distillation")
    parser.add_argument("--teacher_model_path", type=str, default="models_vgg/best_model1.keras", help="Path to pretrained teacher model")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")

    args = parser.parse_args()
    main(args.teacher_model_path, args.epochs)
