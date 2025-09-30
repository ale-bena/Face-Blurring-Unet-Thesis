import cv2
import os
import argparse
from PIL import Image
from BlazeFaceDetection.blazeFaceDetector import blazeFaceDetector

counter=0

#Calculate the optimal number of zeros based on file count
def calculate_optimal_zeros(num_files):
    if num_files <= 0:
        return 1
    return len(str(num_files))

# Blurs faces in the image
def process_image(image, faceDetector):
  global counter
  counter=counter+1

  # Detect faces
  detectionResults = faceDetector.detectFaces(image)

  for box in detectionResults.boxes:
    # Convert normalized coordinates in pixel
      x1 = int(box[0] * image.shape[1])
      y1 = int(box[1] * image.shape[0])
      x2 = int(box[2] * image.shape[1])
      y2 = int(box[3] * image.shape[0])

      # Check if the box area is sufficient(the face shouldn't be too small)
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

# Rename files with zero-padded style
def rename(folder, prefix="img", num_zeros=4, sort_files=False):
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    
    # Sort in alphabetical order if requested
    if sort_files:
        image_files.sort()

    # Create zero-padded number and rename files
    for i, filename in enumerate(image_files, start=1):
        _, ext = os.path.splitext(filename)
        padded_num = str(i).zfill(num_zeros)
        new_name = f"{prefix}{padded_num}{ext}"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        # print(f"renamed: {filename} -> {new_name}") #Only for debugging


def main(input_dir, output_dir, model_type, score_threshold, iou_threshold, sort_files, rename_files, num_zeros, auto_zeros):
    os.makedirs(output_dir, exist_ok=True) # create output folder if doesn't exist
    

    if rename_files:
        if auto_zeros: # Count image files to determine optimal zeros
            image_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
            optimal_zeros = calculate_optimal_zeros(len(image_files))
            print(f"Found {len(image_files)} images, using {optimal_zeros} zeros for numbering")
            actual_zeros = optimal_zeros
        else:
            print(f"Using manual setting: {num_zeros} zeros")
            actual_zeros = num_zeros
        
        rename(input_dir, prefix="img", num_zeros=actual_zeros, sort_files=sort_files)

    # Initialize face detector
    faceDetector = blazeFaceDetector(model_type, score_threshold, iou_threshold)

    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    # Sort files if requested, no operation neede if rename has been done
    if sort_files and not rename_files:  
        image_files.sort()
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image=cv2.imread(input_path, cv2.IMREAD_COLOR)

        if image is not None:
            processed_img = process_image(image, faceDetector)
            #processed_img.save(output_path)
            cv2.imwrite(output_path, processed_img)
            # print(f"Modified image saved: {output_path}") #Only for debugging
        else:
            print(f"Error with {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset blurring")

    parser.add_argument("--input_dir", required=True, help="Path to the dataset with the original images")
    parser.add_argument("--output_dir", required=True, help="Path to the blurred dataset destination")
    parser.add_argument("--score_threshold", type=float, default=0.7, help="score threshold" )
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="iou threshold" )
    parser.add_argument("--model_type", type=str, default="back", help="model type")
    parser.add_argument("--sort", action="store_true", help="Sort files alphabetically before processing")
    parser.add_argument("--rename", action="store_true", help="Rename files with zero-padded numbering")
    parser.add_argument("--num_zeros", type=int, default=4, help="Number of zeros in filename (e.g., 4 for 0001.jpg)")
    parser.add_argument("--auto_zeros", action="store_true", help="Automatically determine number of zeros based on file count")
    
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_type, args.score_threshold, args.iou_threshold, args.sort, args.rename, args.num_zeros, args.auto_zeros)
