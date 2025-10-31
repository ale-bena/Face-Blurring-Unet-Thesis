import cv2
import os
import argparse
from PIL import Image
from BlazeFaceDetection.blazeFaceDetector import blazeFaceDetector

counter = 0

def anonymize_face(img, bbox, scale_factor=0.5, min_kernel=25, max_kernel=101):
    #blur is applied proportionaly to the size of the face
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]

    # Kernel computation
    face_size = max(y2 - y1, x2 - x1)
    blur_kernel = int(face_size * scale_factor)

    if blur_kernel % 2 == 0: # the kernel must be even
        blur_kernel += 1

    blur_kernel = max(min_kernel, min(max_kernel, blur_kernel))

    roi_blurred = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)
    img[y1:y2, x1:x2] = roi_blurred

    return img

def process_image(image, faceDetector):
    global counter
    counter += 1

    detectionResults = faceDetector.detectFaces(image)

    for box in detectionResults.boxes:
        x1 = int(box[0] * image.shape[1])
        y1 = int(box[1] * image.shape[0])
        x2 = int(box[2] * image.shape[1])
        y2 = int(box[3] * image.shape[0])

        image = anonymize_face(image, (x1, y1, x2, y2))
        
    return image

def rename(folder, prefix="img"):
    jpg_files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")] # If png format is used change is needed
    for i, filename in enumerate(jpg_files, start=1):
        new_name = f"{prefix}{i}.jpg"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)

def main(input_dir, output_dir, model_type, score_threshold, iou_threshold):
    os.makedirs(output_dir, exist_ok=True)

    faceDetector = blazeFaceDetector(model_type, score_threshold, iou_threshold)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)

            if image is not None:
                processed_img = process_image(image, faceDetector)
                cv2.imwrite(output_path, processed_img)
                print(f"Modified image saved: {output_path}")
            else:
                print(f"Error with {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset blurring")
    parser.add_argument("--input_dir", required=True, help="Path to the dataset with the original images")
    parser.add_argument("--output_dir", required=True, help="Path to the blurred dataset destination")
    parser.add_argument("--score_threshold", type=float, default=0.7, help="score threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="iou threshold")
    parser.add_argument("--model_type", type=str, default="front", help="model type")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_type, args.score_threshold, args.iou_threshold)

