import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)

#Calculate the optimal number of zeros based on file count
def calculate_optimal_zeros(num_files):
    if num_files <= 0:
        return 1
    return len(str(num_files))

# Rename files with zero-padded style
def rename_files(folder, prefix="", num_zeros=4):
    """Rename files with zero-padded numbering (e.g., 0001.jpg or img0001.jpg)"""
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("No image files found to rename.")
        return

    print(f"Renaming {len(image_files)} files with {num_zeros} zeros...")

    for i, filename in enumerate(image_files, start=1):
        _, ext = os.path.splitext(filename)
        padded_num = str(i).zfill(num_zeros)
        new_name = f"{prefix}{padded_num}{ext}"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
    
    print("File renaming completed!")

def load_and_preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def save_image(tensor, path, output_details=None, format_type='png', quality=95):
    # If the output is quantized --> dequantize
    if output_details and output_details[0]['dtype'] in [np.int8, np.uint8]:
        scale, zero_point = output_details[0]['quantization']
        tensor = tf.cast(tensor, tf.float32)
        tensor = (tensor - zero_point) * scale  # bring back to range [0,1]

    # Clip between 0 and 1 and convert to uint8
    img = tf.clip_by_value(tensor, 0.0, 1.0)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    
    if format_type.lower() == 'jpg' or format_type.lower() == 'jpeg':
        img = tf.image.encode_jpeg(img, quality=quality)
    else:
        img = tf.image.encode_png(img)
    
    tf.io.write_file(path, img)

def detect_model_type(model_path):
    if model_path.lower().endswith('.keras') or model_path.lower().endswith('.h5'):
        return 'keras'
    elif model_path.lower().endswith('.tflite'):
        return 'tflite'
    else:
        raise ValueError(f"Unsupported model format: {model_path}. Use .keras, .h5, or .tflite")

def run_keras_inference(model_path, image_paths, output_dir, format_type, quality):
    print(f"Loading Keras model from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    for idx, img_path in enumerate(image_paths, start=1):
        img = load_and_preprocess_image(img_path)
        img_batch = tf.expand_dims(img, axis=0)
        
        # Inference
        pred = model(img_batch, training=False)
        pred_image = tf.squeeze(pred, axis=0)
        
        # Save
        #save_filename = f"{idx:03}.{format_type}"
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_filename = f"{base_name}.{format_type}"
        save_path = os.path.join(output_dir, save_filename)
        save_image(pred_image, save_path, format_type=format_type, quality=quality)
        print(f"Saved: {save_path}")

def run_tflite_inference(model_path, image_paths, output_dir, format_type, quality):
    print(f"Loading TFLite model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(f"Input shape: {input_details[0]['shape']}")
    # print(f"Input dtype: {input_details[0]['dtype']}")
    # print(f"Output shape: {output_details[0]['shape']}")
    # print(f"Output dtype: {output_details[0]['dtype']}")
    
    for idx, img_path in enumerate(image_paths, start=1):
        img = load_and_preprocess_image(img_path)
        img_batch = tf.expand_dims(img, axis=0)
        input_dtype = input_details[0]['dtype']
    
        input_data = img_batch.numpy()

        if input_dtype == np.int8:
            scale, zero_point = input_details[0]['quantization']
            input_data = input_data / scale + zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        elif input_dtype == np.uint8:
            scale, zero_point = input_details[0]['quantization']
            input_data = input_data / scale + zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)
        else:  # float32
            input_data = input_data.astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details[0]['index'])
        pred_image = tf.squeeze(pred, axis=0)

        # Save
        #save_filename = f"{idx:03}.{format_type}"
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_filename = f"{base_name}.{format_type}"
        save_path = os.path.join(output_dir, save_filename)
        save_image(pred_image, save_path, output_details, format_type, quality)
        print(f"Saved: {save_path}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.rename:
        if args.auto_zeros:
            image_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]
            optimal_zeros = calculate_optimal_zeros(len(image_files))
            print(f"Found {len(image_files)} images, using {optimal_zeros} zeros for numbering")
            actual_zeros = optimal_zeros
        else:
            print(f"Using manual setting: {args.num_zeros} zeros")
            actual_zeros = args.num_zeros
        
        rename_files(args.input_dir, prefix=args.prefix, num_zeros=actual_zeros)
    
    image_paths = [os.path.join(args.input_dir, fname)
                   for fname in os.listdir(args.input_dir)
                   if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_paths)} images to process")
    
    model_type = detect_model_type(args.model_path)
    print(f"Model type detected: {model_type.upper()}")
    
    if model_type == 'keras':
        run_keras_inference(args.model_path, image_paths, args.output_dir, 
                           args.format, args.quality)
    elif model_type == 'tflite':
        run_tflite_inference(args.model_path, image_paths, args.output_dir, 
                            args.format, args.quality)
    print("Inference completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for keras and tflite models")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the model file (.keras, .h5, or .tflite)")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input test images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save output images")
    
    parser.add_argument("--format", type=str, choices=['png', 'jpg', 'jpeg'], default='png',
                       help="Output image format (png or jpg). Default: png")
    parser.add_argument("--quality", type=int, default=95, 
                       help="JPEG quality (1-100, only used for jpg format). Default: 95")
    
    parser.add_argument("--rename", action="store_true", 
                       help="Rename input files with zero-padded numbering before processing")
    parser.add_argument("--num_zeros", type=int, default=4, 
                       help="Number of zeros in filename (e.g., 4 for 0001.jpg). Default: 4")
    parser.add_argument("--auto_zeros", action="store_true", 
                       help="Automatically determine number of zeros based on file count")
    parser.add_argument("--prefix", type=str, default="", 
                       help="Optional prefix for renamed files (e.g., 'img' for img0001.jpg). Default: no prefix")
    
    args = parser.parse_args()
    main(args)
