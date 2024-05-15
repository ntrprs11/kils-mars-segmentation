import yolo as y
import functions as f
import cow as c
from ultralytics import YOLO
import plotly.graph_objects as go
import pandas as pd
import cv2
import shutil
from data_connect.interact_sftp_with_defaults import download_folder_default

# Downloading the folder from the SFTP server to the local machine
download_folder_default(remotepath ="./kils_mars_segmentation", localpath="./train")

def evaluate_all_methods():
    # Setting the path to the test set
    test_path = "dataset/images/test/"
    # Initializing two YOLO models, "model" and "model_x"
    model = YOLO("n-yolo-mars-seg.pt")
    model_x = YOLO("x-yolo-mars-seg.pt")

    # Calculating the Intersection over Union (IoU) for each model
    yolo_mean_iou, yolo_median_iou, yolo_iou = y.evaluate_model(
        model, test_path)
    cow_mean_iou, cow_median_iou, cow_iou = c.evaluate_cow(test_path)
    opencv_mean_iou, opencv_median_iou, opencv_iou = f.evaluate_opencv(
        test_path)
    yolo_x_mean_iou, yolo_x_median_iou, yolo_x_iou = y.evaluate_model(
        model_x, test_path)

    # IoU results for each method
    # yolo: mean Intersection over Union: 0.679925580440089
    # yolo: median Intersection over Union: 0.7000449408375975

    # cow: mean Intersection over Union: 0.14509135131049206
    # cow: median Intersection over Union: 0.1466238125860863

    # opencv: mean Intersection over Union: 0.1536064733316657
    # opencv: median Intersection over Union: 0.08537750137307695

    # yolo_x: mean Intersection over Union: 0.6704760318594339
    # yolo_x: median Intersection over Union: 0.6942458562714163

    # Returning the calculated IoUs for each method
    return yolo_iou, cow_iou, opencv_iou, yolo_x_iou

# Function to display violin plot showing IoU distribution for each method
def export_df_to_csv(yolo_iou, cow_iou, opencv_iou, yolo_x_iou):
    # Creating a DataFrame with IoU values for each method
    df = pd.DataFrame({
        'method': ['cow'] * len(cow_iou) + ['opencv-python'] * len(opencv_iou) + ['YOLO-x'] * len(yolo_x_iou) + ['YOLO-n'] * len(yolo_iou),
        'iou': cow_iou + opencv_iou + yolo_x_iou + yolo_iou
    })
    df.to_csv("iou.csv")


def show_fig():
    df = pd.read_csv("iou.csv")
    # Creating the violin plot
    fig = go.Figure()

    methods = ['cow', 'opencv-python', 'YOLO-x', 'YOLO-n']

    for method in methods:
        fig.add_trace(go.Violin(x=df['method'][df['method'] == method],
                                y=df['iou'][df['method'] == method],
                                name=method,
                                box_visible=True,
                                meanline_visible=True))
    # Displaying the violin plot
    fig.show()

# Function to save images generated by different segmentation methods
def save_image_from_all_three_methods(model, sample_image, true_value, dir):
    results = model.predict(sample_image)
    # Saving the images
    cv2.imwrite(f"{dir}normal.png", sample_image)
    cv2.imwrite(f"{dir}mask.png", true_value*255)
    y.save_mask_image_from_yolo_output(results, f"{dir}yolo.png")
    cv2.imwrite(f"{dir}cow.png", c.predict_cow())
    cv2.imwrite(f"{dir}opencv.png",
                f.predict_sample_using_simple_cv(sample_image))


if __name__ == "__main__":
    # Evaluating the methods and displaying the results
    yolo_iou, cow_iou, opencv_iou, yolo_x_iou = evaluate_all_methods()
    export_df_to_csv(yolo_iou, cow_iou, opencv_iou, yolo_x_iou)
    show_fig()
    # X_train, y_train, X_test, y_test, X_val, y_val = f.load_data()
    # test_path = "dataset/images/test/"
    # model = YOLO("yolo-mars-seg.pt")
    # sample_image, true_value = f.get_sample(6, X_test, y_test) # 65
    # save_image_from_all_three_methods(model, sample_image, true_value, "ps/65/")
    pass