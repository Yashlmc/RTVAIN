import cv2
import os

# Paths
image_dir = "dataset/images/train"
label_dir = "dataset/labels/train"

# Class names (as in your data.yaml)
class_names = ['kicker', 'keeper', 'ball', 'goal_post']

# Output directory
output_dir = "visualized_top_left"
os.makedirs(output_dir, exist_ok=True)

# Loop through all images
for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {img_file}")
        continue

    h, w, _ = img.shape

    if not os.path.exists(label_path):
        print(f"⚠️ No label for {img_file}")
        continue

    # Read label file
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, x_tl, y_tl, bw, bh = map(float, parts)
        cls_id = int(cls_id)

        # Convert normalized top-left to absolute pixel coordinates
        x1 = int(x_tl * w)
        y1 = int(y_tl * h)
        x2 = int((x_tl + bw) * w)
        y2 = int((y_tl + bh) * h)

        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show image
    cv2.imshow("Top-left Box Visualization", img)
    cv2.imwrite(os.path.join(output_dir, img_file), img)

    if cv2.waitKey(500) & 0xFF == 27:  # press ESC to stop early
        break

cv2.destroyAllWindows()