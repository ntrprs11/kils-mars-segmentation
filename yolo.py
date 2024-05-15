from ultralytics import YOLO
import functions as f
import os
import shutil
import numpy as np
import cv2

# Function to check if a pixel is a boundary pixel
def is_boundary_pixel(label, x, y):
    if label[y, x] == 1:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if 0 <= x + dx < label.shape[1] and 0 <= y + dy < label.shape[0]:
                    if label[y + dy, x + dx] == 0:
                        return True
    return False

# Convert annotations from numpy format to YOLO format
def numpy_to_yolo_format(annotations, output_dir):
    for index, label in enumerate(annotations):
        objects = []
        visited = set()
        for y in range(label.shape[0]):
            for x in range(label.shape[1]):
                if (x, y) not in visited and label[y, x] == 1:

                    object_pixels = []
                    boundary_pixels = [(x, y)]
                    while boundary_pixels:
                        px, py = boundary_pixels.pop()
                        if (px, py) not in visited and label[py, px] == 1:
                            object_pixels.append((px, py))
                            visited.add((px, py))

                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    nx, ny = px + dx, py + dy
                                    if (nx, ny) not in visited and 0 <= nx < label.shape[1] and 0 <= ny < label.shape[0]:
                                        boundary_pixels.append((nx, ny))
                    objects.append(object_pixels)

        yolo_format = []
        for obj in objects:
            obj_str = "0 "
            for px, py in obj:
                obj_str += f"{px / 255} {py / 255} "
            yolo_format.append(obj_str.strip())

        with open(f"{output_dir}/{index}.txt", "w") as txt:
            for line in yolo_format:
                txt.write(line + "\n")

# Ensure directory exists or create it
def make_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

# Move files to YOLO directory structure
def move_files_to_yolo_dir(image_dir, label_dir, train_test_val):
    make_dir_if_not_exist("./dataset/")
    make_dir_if_not_exist("./dataset/images/")
    make_dir_if_not_exist("./dataset/labels/")
    make_dir_if_not_exist("./dataset/images/train/")
    make_dir_if_not_exist("./dataset/images/val/")
    make_dir_if_not_exist("./dataset/images/test/")
    make_dir_if_not_exist("./dataset/labels/train/")
    make_dir_if_not_exist("./dataset/labels/val/")
    make_dir_if_not_exist("./dataset/labels/test/")

    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    sample_len = len(images)
    for idx, (image, label) in enumerate(zip(images, labels)):
        if idx / sample_len < train_test_val[0]:
            shutil.copy(image_dir + image, "./dataset/images/train/" + image)
            shutil.copy(label_dir + label, "./dataset/labels/train/" + label)
        elif train_test_val[0] <= idx / sample_len < train_test_val[1]:
            shutil.copy(image_dir + image, "./dataset/images/test/" + image)
            shutil.copy(label_dir + label, "./dataset/labels/test/" + label)
        else:
            shutil.copy(image_dir + image, "./dataset/images/val/" + image)
            shutil.copy(label_dir + label, "./dataset/labels/val/" + label)

# Save mask image from YOLO output
def save_mask_image_from_yolo_output(results, path=None):
    image = np.zeros((256, 256), dtype=np.uint8)

    for result in results:
        mask = result.masks.xy

        for obj in mask:
            xy_poly = []

            for xy in obj:
                y = int(xy[1])
                x = int(xy[0])
                xy_poly.append([x, y])

            image = cv2.fillPoly(image, [np.array(xy_poly)], 255)

    if path is not None:
        cv2.imwrite(path, image)

    return image

# Train YOLO model
def train_yolo(data, epochs):
    model = YOLO("yolov8x-seg.pt")
    model.train(data=data, epochs=epochs, patience=10, imgsz=256,
                cos_lr=True, pretrained="yolov8x-seg.pt")

# Predict using YOLO model
def predict_yolo(model, image, save=True):
    results = model.predict(image, save=save)
    return results

# Evaluate YOLO model
def evaluate_model(model, test_img_dir):
    iou = []
    temp_path = "./temp/"
    make_dir_if_not_exist(temp_path)

    test_img_paths = os.listdir(test_img_dir)
    for test_img in test_img_paths:
        results = predict_yolo(model, test_img_dir + test_img, save=False)
        image_idx = test_img.split(sep=".")[0]
        yolo_name = "yolo_" + image_idx + ".png"
        _ = save_mask_image_from_yolo_output(results, temp_path + yolo_name)
        mask_img = "mask_" + image_idx + ".png"
        shutil.copy("train_mask/" + mask_img, temp_path + mask_img)

        mask = cv2.imread(temp_path + mask_img)
        yolo = cv2.imread(temp_path + yolo_name)

        iou.append(f.intersection_over_union(mask, yolo))

        os.remove(temp_path + mask_img)
        os.remove(temp_path + yolo_name)

    mean_iou = np.mean(iou)
    median_iou = np.median(iou)

    print(f"yolo: mean Intersection over Union: {mean_iou}")
    print(f"yolo: median Intersection over Union: {median_iou}")

    return mean_iou, median_iou, iou


if __name__ == "__main__":
    # annotations = f.load_annotations()
    # numpy_to_yolo_format(annotations, "train_labels")

    # move_files_to_yolo_dir("train/", "train_labels/", [0.85, 0.925])
    train_yolo("dataset.yaml", 500)
    # image_path = r"dataset\images\test\6.png"
    # model = YOLO("yolo-mars-seg.pt")
    # results = predict_yolo(model, r"dataset\images\test\6.png", False)
    # _ = save_mask_image_from_yolo_output(results, "mask_6.png")

    # evaluate_model(model, "dataset/images/test/")
