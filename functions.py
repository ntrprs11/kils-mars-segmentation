import numpy as np
import cv2
import os
import copy
from collections import Counter
import yolo as y
import shutil
import functions as f


# function to load annotations from a numpy file
def load_annotations() -> np.ndarray:
    annotations = np.unpackbits(np.load('train.npy'))
    return annotations.reshape((6500, 256, 256))


# function to save annotations as mask images
def save_annotations_as_masks():
    annotations = load_annotations()
    for i in range(len(annotations)):
        image = annotations[i] * 255
        image = image.astype(np.uint8)
        cv2.imwrite(f"train_mask/mask_{i}.png", image)


# function to load images from a directory
def load_images(dir: str):
    images = []
    dir_list = os.listdir(dir)
    sorted_dir = sorted(dir_list, key=lambda x: int(x.split('.')[0]))
    for file in sorted_dir:
        images.append(cv2.imread(os.path.join(dir, file)))

    return np.array(images)


# function to append corresponding y values to a list
def append_y(images_path, y, list):
    image_paths = os.listdir(images_path)
    images_sorted = sorted(image_paths, key=lambda x: int(x.split('.')[0]))
    idxs = []
    for image in images_sorted:
        idxs.append(int(image.split(sep=".")[0]))

    for idx in idxs:
        real_idx = idx
        list.append(y[real_idx])

    return list


# function to load data
def load_data():
    X_train_path = "dataset/images/train/"
    X_val_path = "dataset/images/val/"
    X_test_path = "dataset/images/test/"

    print("Loading images...\n")
    X_train = load_images(X_train_path)
    X_val = load_images(X_val_path)
    X_test = load_images(X_test_path)

    print("Loading labels...\n")
    y = load_annotations()

    y_train = []
    y_val = []
    y_test = []

    y_train = append_y(X_train_path, y, y_train)
    y_val = append_y(X_val_path, y, y_val)
    y_test = append_y(X_test_path, y, y_test)

    return X_train, np.array(y_train), X_test, np.array(y_test), X_val, np.array(y_val)


# function to get a sample image and its true value
def get_sample(sample, X_test, y_test):
    sample_image = X_test[sample]
    true_value = y_test[sample]

    return sample_image, true_value


# function to calculate IoU between two images
def intersection_over_union(image1, image2):
    # Convert images to binary masks based on threshold
    mask1 = image1
    mask2 = image2

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0.0
    return iou


# function for grid search to find optimal parameters
def grid_search(X_test, y_test, morph, gaussian, expand_radius):
    performance_overall = {
        "morphs": [],
        "gaussians": [],
        "expand_radius": [],
    }

    calculations = len(morph) * len(gaussian) * \
        len(expand_radius) * X_test.shape[0]
    count = 1
    for X, y in zip(X_test, y_test):
        performance = {
            "morph": [],
            "gaussian": [],
            "expand_radius": [],
            "iou": []
        }
        for m in morph:
            for g in gaussian:
                for er in expand_radius:
                    print(f"Calculating {count}/{calculations}")
                    result_image = predict_sample_using_simple_cv(
                        X, morph=m, gaussian=g, expand_radius=er)
                    iou = intersection_over_union(y, result_image)
                    performance["morph"].append(m)
                    performance["gaussian"].append(g)
                    performance["expand_radius"].append(er)
                    performance["iou"].append(iou)
                    count += 1

        iou_values = np.array(performance["iou"])
        min_index = np.argmax(iou_values)
        performance_overall["morphs"].append(performance["morph"][min_index])
        performance_overall["gaussians"].append(
            performance["gaussian"][min_index])
        performance_overall["expand_radius"].append(
            performance["expand_radius"][min_index])

    most_common_values = {}

    for key, values in performance_overall.items():
        counter = Counter(values)
        most_common_values[key] = counter.most_common(1)[0][0]

    best_morph = most_common_values["morphs"]
    best_gaussian = most_common_values["gaussians"]
    best_expand_radius = most_common_values["expand_radius"]

    return best_morph, best_gaussian, best_expand_radius


# function to perform the grid search and print the best parameters
def perform_grid_search(X_test, y_test):
    morph = []
    for i in range(5):
        value = 3 + i*2
        morph.append((value, value))
    gaussian = []
    for j in range(20):
        value = j*2 + 1
        gaussian.append((value, value))
    expand_radius = []
    for k in range(5):
        expand_radius.append(5 + k*5)

    best_morph, best_gaussian, best_expand_radius = grid_search(
        X_test, y_test, morph=morph, gaussian=gaussian, expand_radius=expand_radius)
    print(
        f"morph: {best_morph}\ngaussian: {best_gaussian}\ner: {best_expand_radius}")
    # Calculating 2762500/2762500
    # morph: (3, 3)
    # gaussian: (39, 39)
    # er: 5


# function to expand black regions in an image
def expand_black(image, factor):
    inverted_image = cv2.bitwise_not(image)

    kernel = np.ones((factor, factor), np.uint8)

    dilated_image = cv2.dilate(inverted_image, kernel, iterations=1)

    expanded_image = cv2.bitwise_not(dilated_image)

    return expanded_image


# function to predict sample using simple CV operations
def predict_sample_using_simple_cv(X_test, sample=None, morph=(3, 3), gaussian=(15, 15), expand_radius=20):
    if sample:
        sample_image = X_test[sample]
    else:
        sample_image = X_test

    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    # 37, 37
    # applying gaussian blur
    sample_image = cv2.GaussianBlur(sample_image, gaussian, 0)
    _, flat_mask = cv2.threshold(
        sample_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones(morph, np.uint8)
    # applying morphological opening
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN, kernel)

    expanded = expand_black(flat_mask, expand_radius)
    return expanded


# function to evaluate OpenCV method
def evaluate_opencv(test_img_dir):
    iou = []
    temp_path = "./temp/"
    y.make_dir_if_not_exist(temp_path)

    test_img_paths = os.listdir(test_img_dir)
    for test_img in test_img_paths:
        image = cv2.imread(test_img_dir + test_img)
        image = predict_sample_using_simple_cv(image)
        image_idx = test_img.split(sep=".")[0]
        opencv_name = "opencv_" + image_idx + ".png"
        cv2.imwrite(temp_path + opencv_name, image)
        mask_img = "mask_" + image_idx + ".png"
        shutil.copy("train_mask/" + mask_img, temp_path + mask_img)

        mask = cv2.imread(temp_path + mask_img)
        opencv = cv2.imread(temp_path + opencv_name)

        iou.append(f.intersection_over_union(mask, opencv))

        os.remove(temp_path + mask_img)
        os.remove(temp_path + opencv_name)

    mean_iou = np.mean(iou)
    median_iou = np.median(iou)

    print(f"opencv: mean Intersection over Union: {mean_iou}")
    print(f"opencv: median Intersection over Union: {median_iou}")

    return mean_iou, median_iou, iou


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    # image = predict_sample_using_simple_cv(X_test, 1)
    # cv2.imwrite("predicted.png", image)

    evaluate_opencv("dataset/images/test/")
    # perform_grid_search(X_train, y_train)