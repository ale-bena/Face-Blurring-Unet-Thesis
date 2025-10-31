import cv2
import os
import argparse
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
counter = 0


def calculate_optimal_zeros(num_files):
    if num_files <= 0:
        return 1
    return len(str(num_files))


def anonymize_face(img, bbox, scale_factor=0.5, min_kernel=25, max_kernel=101):
    """
    Apply blur proportional to face size.
    scale_factor: fraction of face size to determine kernel
    min_kernel, max_kernel: lower and upper kernel limits (must be odd)
    """
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]

    face_size = max(y2 - y1, x2 - x1)
    blur_kernel = int(face_size * scale_factor)

    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_kernel = max(min_kernel, min(max_kernel, blur_kernel))

    roi_blurred = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)
    img[y1:y2, x1:x2] = roi_blurred

    return img


def process_image(image, face_detector):
    global counter
    counter += 1

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if results.detections:
        h, w, _ = image.shape
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)
            image = anonymize_face(image, (x1, y1, x2, y2))

    return image


def rename(folder, prefix="img", num_zeros=5, sort_files=False):
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]

    if sort_files:
        image_files.sort()

    for i, filename in enumerate(image_files, start=1):
        _, ext = os.path.splitext(filename)
        padded_num = str(i).zfill(num_zeros)
        new_name = f"{prefix}{padded_num}{ext}"
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))


def main(input_dir, output_dir, sort_files, rename_files, num_zeros, auto_zeros):
    os.makedirs(output_dir, exist_ok=True)

    if rename_files:
        if auto_zeros:
            image_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
            optimal_zeros = calculate_optimal_zeros(len(image_files))
            print(f"Found {len(image_files)} images, using {optimal_zeros} zeros for numbering")
            actual_zeros = optimal_zeros
        else:
            actual_zeros = num_zeros
            print(f"Using manual setting: {num_zeros} zeros")

        rename(input_dir, prefix="img", num_zeros=actual_zeros, sort_files=sort_files)

    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    if sort_files and not rename_files:
        image_files.sort()

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Error loading {filename}")
                continue

            processed_img = process_image(image, face_detector)
            cv2.imwrite(output_path, processed_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset blurring with MediaPipe (dynamic blur)")
    parser.add_argument("--input_dir", required=True, help="Path to the original dataset")
    parser.add_argument("--output_dir", required=True, help="Path to save blurred dataset")
    parser.add_argument("--sort", action="store_true", help="Sort files alphabetically")
    parser.add_argument("--rename", action="store_true", help="Rename files with zero padding")
    parser.add_argument("--num_zeros", type=int, default=5, help="Number of zeros for file naming")
    parser.add_argument("--auto_zeros", action="store_true", help="Auto-determine zero padding based on count")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.sort, args.rename, args.num_zeros, args.auto_zeros)
