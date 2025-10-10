import os
import argparse
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from model_teacher import build_blur_unet as build_teacher 
from model_student import build_blur_unet as build_student

DEFAULT_IMG_SIZE = (128, 128) 
DEFAULT_BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

@register_keras_serializable()
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

@register_keras_serializable()
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def process_path(img_path, tgt_path, img_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    tgt = tf.io.read_file(tgt_path)
    tgt = tf.image.decode_png(tgt, channels=3)
    tgt = tf.image.resize(tgt, img_size)
    tgt = tf.cast(tgt, tf.float32) / 255.0

    return img, tgt

def load_dataset(image_dir, target_dir, img_size, batch_size, max_images=None):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".jpg")])

    if max_images is not None:
        image_files = image_files[:max_images]
        target_files = target_files[:max_images]

    image_ds = tf.data.Dataset.from_tensor_slices(image_files)
    target_ds = tf.data.Dataset.from_tensor_slices(target_files)

    dataset = tf.data.Dataset.zip((image_ds, target_ds))
    dataset = dataset.map(
        lambda x, y: process_path(x, y, img_size),
        num_parallel_calls=AUTOTUNE,
        deterministic=True
    )
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

@register_keras_serializable()
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, alpha=0.7, beta=0.3, **kwargs):
        super(Distiller, self).__init__(**kwargs)
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "student": self.student,
            "teacher": self.teacher,
        })
        return config

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.optimizer = optimizer

    def train_step(self, data):
        x, y_true = data

        # Teacher inference --> not updated)
        y_teacher = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            y_student = self.student(x, training=True)

            # Loss 1: student vs ground-truth
            loss_gt = tf.reduce_mean(tf.square(y_true - y_student))

            # Loss 2: student vs teacher
            loss_distill = tf.reduce_mean(tf.square(y_teacher - y_student))

            # Loss total = weighted combination
            loss = self.alpha * loss_gt + self.beta * loss_distill

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        # Update metrics (respect to ground-truth)
        for metric in self.metrics:
            metric.update_state(y_true, y_student)

        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        results["loss_gt"] = loss_gt
        results["loss_distill"] = loss_distill
        return results

    def test_step(self, data):
        x, y_true = data
        y_teacher = self.teacher(x, training=False)
        y_student = self.student(x, training=False)

        loss_gt = tf.reduce_mean(tf.square(y_true - y_student))
        loss_distill = tf.reduce_mean(tf.square(y_teacher - y_student))
        loss = self.alpha * loss_gt + self.beta * loss_distill

        for metric in self.metrics:
            metric.update_state(y_true, y_student)

        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        results["loss_gt"] = loss_gt
        results["loss_distill"] = loss_distill
        return results

def train_model(teacher_model_path, epochs, img_size, batch_size, alpha, beta, 
                train_images_dir, train_targets_dir, val_images_dir, val_targets_dir,
                max_train_images, max_val_images, output_dir, best_model_name, 
                final_model_name, csv_log_name, resume_training=False, student_path=None):
    print("Loading pretrained Teacher model...")
    teacher = tf.keras.models.load_model(teacher_model_path)
    teacher.trainable = False

    if resume_training and student_path and os.path.exists(student_path):
        print(f"Resuming Student training from: {student_path}")
        student = tf.keras.models.load_model(student_path)
    else:
        print("Building new Student model...")
        student = build_student(input_shape=(*img_size, 3))

    distiller = Distiller(student=student, teacher=teacher, alpha=alpha, beta=beta)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', ssim_metric, psnr_metric]
    )

    print("Loading datasets...")
    train_dataset = load_dataset(train_images_dir, train_targets_dir, img_size, batch_size, max_train_images)
    val_dataset = load_dataset(val_images_dir, val_targets_dir, img_size, batch_size, max_val_images)

    steps_per_epoch = max_train_images // batch_size
    validation_steps = max_val_images // batch_size

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, best_model_name)
    final_model_path = os.path.join(output_dir, final_model_name)
    csv_log_path = os.path.join(output_dir, csv_log_name)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        best_model_path, save_best_only=True, monitor="val_loss", mode="min"
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, mode="min", start_from_epoch=18
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.7, patience=8, min_lr=1e-6, verbose=1, mode="min", start_from_epoch=14
    )
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, append=True)

    print("Starting Student training with MSE loss...")
    history = distiller.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb, csv_logger]
    )

    student.save(final_model_path)
    print(f"Student model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path}")
    print(f"Training log saved to {csv_log_path}")

def main(args):
    print(f"Training student with teacher at: {args.teacher_model_path}")

    if args.gpu_growth:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(e)

    gpus = tf.config.list_physical_devices('GPU')
    for i, gpu in enumerate(gpus):
        try:
            memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
            total_memory = memory_info['peak'] / (1024**3)
            print(f"GPU {i}: {gpu.name}, peak memory: {total_memory:.2f} GB")
        except:
            print(f"GPU {i}: {gpu.name}")

    train_model(
        teacher_model_path=args.teacher_model_path,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        train_images_dir=args.train_images_dir,
        train_targets_dir=args.train_targets_dir,
        val_images_dir=args.val_images_dir,
        val_targets_dir=args.val_targets_dir,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        output_dir=args.output_dir,
        best_model_name=args.best_model_name,
        final_model_name=args.final_model_name,
        csv_log_name=args.csv_log_name,
        resume_training=args.resume_training,
        student_path=args.student_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Student Model with Knowledge Distillation")
    
    # Model and training parameters
    parser.add_argument("--teacher_model_path", type=str, default="models_vgg/best_modelcsvlrf.keras", 
                       help="Path to pretrained teacher model")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--resume_training", action="store_true", 
                       help="Resume training from a saved student model")
    parser.add_argument("--student_path", type=str, default=None, 
                       help="Path to student model to resume training")
    
    # Distillation parameters
    parser.add_argument("--alpha", type=float, default=0.7, 
                       help="Weight for ground-truth loss (default: 0.7)")
    parser.add_argument("--beta", type=float, default=0.3, 
                       help="Weight for distillation loss (default: 0.3)")
    
    # Data parameters
    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128], 
                       help="Input image size (height width). Default: 128 128")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for training. Default: 32")
    
    # Dataset paths
    parser.add_argument("--train_images_dir", type=str, default="./vggface/train", 
                       help="Path to training images directory")
    parser.add_argument("--train_targets_dir", type=str, default="./vggface/train_blur", 
                       help="Path to training targets (blurred) directory")
    parser.add_argument("--val_images_dir", type=str, default="./vggface/val", 
                       help="Path to validation images directory")
    parser.add_argument("--val_targets_dir", type=str, default="./vggface/val_blur", 
                       help="Path to validation targets (blurred) directory")
    
    # Dataset size limits
    parser.add_argument("--max_train_images", type=int, default=8000, 
                       help="Maximum number of training images to use")
    parser.add_argument("--max_val_images", type=int, default=2000, 
                       help="Maximum number of validation images to use")
    
    # GPU configuration
    parser.add_argument("--gpu_growth", action="store_true", 
                       help="Enable GPU memory growth (recommended)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="models_student", 
                       help="Output directory for saving models and logs")
    parser.add_argument("--best_model_name", type=str, default="best_student.keras", 
                       help="Name for the best model checkpoint")
    parser.add_argument("--final_model_name", type=str, default="student_final.keras", 
                       help="Name for the final model")
    parser.add_argument("--csv_log_name", type=str, default="training_log.csv", 
                       help="Name for the CSV training log file")

    args = parser.parse_args()
    main(args)


