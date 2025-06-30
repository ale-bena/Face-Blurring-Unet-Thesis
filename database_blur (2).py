import cv2
import os
import argparse
from PIL import Image
from BlazeFaceDetection.blazeFaceDetector import blazeFaceDetector

counter=0

def process_image(image, faceDetector):
  global counter
  counter=counter+1

  # Detect faces
  detectionResults = faceDetector.detectFaces(image)

  for box in detectionResults.boxes:
    # Converti le coordinate normalizzate in pixel
      x1 = int(box[0] * image.shape[1])
      y1 = int(box[1] * image.shape[0])
      x2 = int(box[2] * image.shape[1])
      y2 = int(box[3] * image.shape[0])

      # Verify that the box area is sufficient
      if (x2 - x1 > 5) and (y2 - y1 > 5):
          sub_face = image[y1:y2, x1:x2]
          if sub_face is not None and sub_face.size != 0:
            blurred = cv2.GaussianBlur(sub_face, (41, 41), 0)
            #blurred = cv2.blur(sub_face, (18, 18))
            image[y1:y2, x1:x2] = blurred
          else:
            print(f"Invalid face image {counter}")
      else:
          print("Box is too small")
  return image


def rename(folder, prefix="img"):
    # Elenco di tutti i file .jpg (case-insensitive)
    jpg_files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    
    # Ordina alfabeticamente (o cambia criterio se vuoi ordine per data)
    #jpg_files.sort()

    for i, filename in enumerate(jpg_files, start=1):
        new_name = f"{prefix}{i}.jpg"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        # Se il nuovo nome esiste già, lo salta o lo gestisce (qui: sovrascrive)
        os.rename(old_path, new_path)
        # print(f"renamed: {filename} -> {new_name}")


def main(input_dir, output_dir, model_type, score_threshold, iou_threshold):
    # create output folder if doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # rename(input_dir)

    # Initialize face detector
    faceDetector = blazeFaceDetector(model_type, score_threshold, iou_threshold)

    # Itera su tutti i file nella cartella
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image=cv2.imread(input_path, cv2.IMREAD_COLOR)

            if image is not None:
                  processed_img = process_image(image, faceDetector)
                  #processed_img.save(output_path)
                  cv2.imwrite(output_path, processed_img)
                  print(f"Modified image saved: {output_path}")
            else:
                print(f"Error with {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset blurring")

    parser.add_argument("--input_dir", required=True, help="Path to the dataset with the original images")
    parser.add_argument("--output_dir", required=True, help="Path to the blurred dataset destination")
    parser.add_argument("--n_avg", type=int, default=18, help="Number of pixel used in the average for blurring, a greater number means more blurring" )
    parser.add_argument("--score_threshold", type=float, default=0.7, help="score threshold" )
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="iou threshold" )
    parser.add_argument("--model_type", type=str, default="back", help="model type")
    
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_type, args.score_threshold, args.iou_threshold)
