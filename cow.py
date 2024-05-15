import numpy as np
import random
import os
import yolo as y
import cv2
import shutil
import functions as f

# function to predict a cow patch
def predict_cow():
    cow = np.load("cow.npy")

    # randomly flip the cow patch along axis 0 and axis 1
    flip_axis_0 = bool(random.getrandbits(1))
    flip_axis_1 = bool(random.getrandbits(1))

    if flip_axis_0:
        cow = np.flip(cow, axis=0)
    if flip_axis_1:
        cow = np.flip(cow, axis=1)

    # randomly scale the cow patch
    scale_x = int(int(cow.shape[0]) * random.uniform(0.75, 1.25))
    scale_y = int(int(cow.shape[1]) * random.uniform(0.75, 1.25))

    cow = np.resize(cow, (scale_x, scale_y))

    # extract a random patch from the cow patch
    height, width = cow.shape[:2]
    patch_size = (256, 256)
    start_x = np.random.randint(0, width - patch_size[1] + 1)
    start_y = np.random.randint(0, height - patch_size[0] + 1)

    patch = cow[start_y:start_y + patch_size[0],
                start_x:start_x + patch_size[1]]

    return patch

# function to evaluate the cow prediction
def evaluate_cow(test_img_dir):
    iou = []
    temp_path = "./temp/"
    y.make_dir_if_not_exist(temp_path)

    test_img_paths = os.listdir(test_img_dir)
    for test_img in test_img_paths:
        # predict cow patch
        patch = predict_cow()
        image_idx = test_img.split(sep=".")[0]
        cow_name = "cow_" + image_idx + ".png"
        cv2.imwrite(temp_path + cow_name, patch)
        mask_img = "mask_" + image_idx + ".png"
        shutil.copy("train_mask/" + mask_img, temp_path + mask_img)

        mask = cv2.imread(temp_path + mask_img)
        cow = cv2.imread(temp_path + cow_name)

        # calculate IoU
        iou.append(f.intersection_over_union(mask, cow))

        os.remove(temp_path + mask_img)
        os.remove(temp_path + cow_name)

    mean_iou = np.mean(iou)
    median_iou = np.median(iou)

    print(f"cow: mean Intersection over Union: {mean_iou}")
    print(f"cow: median Intersection over Union: {median_iou}")

    return mean_iou, median_iou, iou


if __name__ == "__main__":
    evaluate_cow("dataset/images/test/")
