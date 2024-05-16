import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cv2
import yolo as y
from ultralytics import YOLO
import cow as c
import functions as f
# streamlit run streamlit.py

def show_fig():
    df = pd.read_csv("iou.csv")
    
    fig = go.Figure()

    methods = ['cow', 'opencv-python', 'YOLO-x', 'YOLO-n']

    for method in methods:
        fig.add_trace(go.Violin(x=df['method'][df['method'] == method],
                                y=df['iou'][df['method'] == method],
                                name=method,
                                box_visible=True,
                                meanline_visible=True))
    
    return fig

def predict_and_draw(image):
    model = YOLO("n-yolo-mars-seg.pt")
    results = model.predict(image)
    image_mask = y.save_mask_image_from_yolo_output(results)
    return image_mask
    
st.title("KI LifeSciences: Mars Segmentation")
st.markdown("""
made by Robin Oliver Korn and David Deml
### Project Overview
This project aims to segment landable areas in drone images from the mars surface, the dataset was used in the Tensor Tournament 2024.
""")

st.markdown("""
### Example images and masks
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.image("train/0.png", caption="0.png", use_column_width=True)
    st.image("train_mask/mask_0.png", caption="mask_0.png", use_column_width=True)
with col2:
    st.image("train/1.png", caption="1.png", use_column_width=True)
    st.image("train_mask/mask_1.png", caption="mask_1.png", use_column_width=True)
with col3:
    st.image("train/2.png", caption="2.png", use_column_width=True)
    st.image("train_mask/mask_2.png", caption="mask_2.png", use_column_width=True)

test_images = ['5971.png', '5972.png', '5973.png', '5974.png', '5975.png', '5976.png', '5977.png', '5978.png', '5979.png', '598.png', '5980.png', '5981.png', '5982.png', '5983.png', '5984.png', '5985.png', '5986.png', '5987.png', '5988.png', '5989.png', '599.png', '5990.png', '5991.png', '5992.png', '5993.png', '5994.png', '5995.png', '5996.png', '5997.png', '5998.png', '5999.png', '6.png', '60.png', '600.png', '6000.png', '6001.png', '6002.png', '6003.png', '6004.png', '6005.png', '6006.png', '6007.png', '6008.png', '6009.png', '601.png', '6010.png', '6011.png', '6012.png', '6013.png', '6014.png', '6015.png', '6016.png', '6017.png', '6018.png', '6019.png', '602.png', '6020.png', '6021.png', '6022.png', '6023.png', '6024.png', '6025.png', '6026.png', '6027.png', '6028.png', '6029.png', '603.png', '6030.png', '6031.png', '6032.png', '6033.png', '6034.png', '6035.png', '6036.png', '6037.png', '6038.png', '6039.png', '604.png', '6040.png', '6041.png', '6042.png', '6043.png', '6044.png', '6045.png', '6046.png', '6047.png', '6048.png', '6049.png', '605.png', '6050.png', '6051.png', '6052.png', '6053.png', '6054.png', '6055.png', '6056.png', '6057.png', '6058.png', '6059.png', '606.png', '6060.png', '6061.png', '6062.png', '6063.png', '6064.png', '6065.png', '6066.png', '6067.png', '6068.png', '6069.png', '607.png', '6070.png', '6071.png', '6072.png', '6073.png', '6074.png', '6075.png', '6076.png', '6077.png', '6078.png', '6079.png', '608.png', '6080.png', '6081.png', '6082.png', '6083.png', '6084.png', '6085.png', '6086.png', '6087.png', '6088.png', '6089.png', '609.png', '6090.png', '6091.png', '6092.png', '6093.png', '6094.png', '6095.png', '6096.png', '6097.png', '6098.png', '6099.png', '61.png', '610.png', '6100.png', '6101.png', '6102.png', '6103.png', '6104.png', '6105.png', '6106.png', '6107.png', '6108.png', '6109.png', '611.png', '6110.png', '6111.png', '6112.png', '6113.png', '6114.png', '6115.png', '6116.png', '6117.png', '6118.png', '6119.png', '612.png', '6120.png', '6121.png', '6122.png', '6123.png', '6124.png', '6125.png', '6126.png', '6127.png', '6128.png', '6129.png', '613.png', '6130.png', '6131.png', '6132.png', '6133.png', '6134.png', '6135.png', '6136.png', '6137.png', '6138.png', '6139.png', '614.png', '6140.png', '6141.png', '6142.png', '6143.png', '6144.png', '6145.png', '6146.png', '6147.png', '6148.png', '6149.png', '615.png', '6150.png', '6151.png', 
        '6152.png', '6153.png', '6154.png', '6155.png', '6156.png', '6157.png', '6158.png', '6159.png', '616.png', '6160.png', '6161.png', '6162.png', '6163.png', '6164.png', '6165.png', '6166.png', '6167.png', '6168.png', '6169.png', '617.png', '6170.png', '6171.png', '6172.png', '6173.png', '6174.png', '6175.png', '6176.png', '6177.png', '6178.png', '6179.png', '618.png', '6180.png', '6181.png', '6182.png', '6183.png', '6184.png', '6185.png', '6186.png', '6187.png', '6188.png', '6189.png', '619.png', '6190.png', '6191.png', '6192.png', '6193.png', '6194.png', '6195.png', '6196.png', '6197.png', '6198.png', '6199.png', '62.png', '620.png', '6200.png', '6201.png', '6202.png', '6203.png', '6204.png', '6205.png', '6206.png', '6207.png', '6208.png', '6209.png', '621.png', '6210.png', '6211.png', '6212.png', '6213.png', '6214.png', '6215.png', '6216.png', '6217.png', '6218.png', '6219.png', '622.png', '6220.png', '6221.png', '6222.png', '6223.png', '6224.png', '6225.png', '6226.png', '6227.png', '6228.png', '6229.png', '623.png', '6230.png', '6231.png', '6232.png', '6233.png', '6234.png', '6235.png', '6236.png', '6237.png', '6238.png', '6239.png', '624.png', '6240.png', '6241.png', '6242.png', '6243.png', '6244.png', '6245.png', '6246.png', '6247.png', '6248.png', '6249.png', '625.png', '6250.png', '6251.png', '6252.png', '6253.png', '6254.png', '6255.png', '6256.png', '6257.png', '6258.png', '6259.png', '626.png', '6260.png', '6261.png', '6262.png', '6263.png', '6264.png', '6265.png', '6266.png', '6267.png', '6268.png', '6269.png', '627.png', '6270.png', '6271.png', '6272.png', '6273.png', '6274.png', '6275.png', '6276.png', '6277.png', '6278.png', '6279.png', '628.png', '6280.png', '6281.png', '6282.png', '6283.png', '6284.png', '6285.png', '6286.png', '6287.png', '6288.png', '6289.png', '629.png', '6290.png', '6291.png', '6292.png', '6293.png', '6294.png', '6295.png', '6296.png', '6297.png', '6298.png', '6299.png', '63.png', 
        '630.png', '6300.png', '6301.png', '6302.png', '6303.png', '6304.png', '6305.png', '6306.png', '6307.png', '6308.png', '6309.png', '631.png', '6310.png', '6311.png', '6312.png', '6313.png', '6314.png', '6315.png', '6316.png', '6317.png', '6318.png', '6319.png', '632.png', '6320.png', '6321.png', '6322.png', '6323.png', '6324.png', '6325.png', '6326.png', '6327.png', '6328.png', '6329.png', '633.png', '6330.png', '6331.png', '6332.png', '6333.png', '6334.png', '6335.png', '6336.png', '6337.png', '6338.png', '6339.png', '634.png', '6340.png', '6341.png', '6342.png', '6343.png', '6344.png', '6345.png', '6346.png', '6347.png', '6348.png', '6349.png', '635.png', '6350.png', '6351.png', '6352.png', '6353.png', '6354.png', '6355.png', '6356.png', 
        '6357.png', '6358.png', '6359.png', '636.png', '6360.png', '6361.png', '6362.png', '6363.png', '6364.png', '6365.png', '6366.png', '6367.png', '6368.png', '6369.png', '637.png', '6370.png', '6371.png', '6372.png', '6373.png', '6374.png', '6375.png', '6376.png', '6377.png', '6378.png', '6379.png', '638.png', '6380.png', '6381.png', '6382.png', '6383.png', '6384.png', '6385.png', '6386.png', '6387.png', '6388.png', '6389.png', '639.png', '6390.png', '6391.png', '6392.png', '6393.png', '6394.png', '6395.png', '6396.png', '6397.png', '6398.png', '6399.png', '64.png', '640.png', '6400.png', '6401.png', '6402.png', '6403.png', '6404.png', '6405.png', '6406.png', '6407.png', '6408.png', '6409.png'] 


st.markdown("""
### Technical Details
We used pretrained [YOLO](https://docs.ultralytics.com/) segmentation models for the segmentation. We had to fine tune the models to our dataset.
""")

st.markdown("""
### Method Performance & Violin Plot
We evaluated our methods using the Jaccard Index (Intersection over Union) with the true labels.
Feel free to explore the plot!  
""")
violin_plot = show_fig()
st.plotly_chart(violin_plot)

st.markdown("""
### Try predicting an Image yourself using our YOLO-n model!
""")

selected_image = st.selectbox("Use the Dropdown to select an image and click on the Predict Button below!", test_images)
image_path = "train/" + selected_image
image = image_path
st.image(image, caption=selected_image, use_column_width=True)

_, _, _, col, _, _, _ = st.columns(7)
with col:
    predict = st.button("Predict")

if predict:
    st.markdown("""
### Try predicting an Image yourself using our methods!
Left: n-YOLO, Middle: opencv-python, Rigth: cow algorithm
""")

selected_image = st.selectbox("Use the Dropdown to select an image and click on the Predict Button below!", test_images)
image_path = "train/" + selected_image
image = image_path
st.image(image, caption=selected_image, use_column_width=True)

_, _, _, col, _, _, _ = st.columns(7)
with col:
    predict = st.button("Predict")

if predict:
    coll, colm, colr = st.columns(3)
    with coll:
        n_result_image = predict_and_draw("n-yolo-mars-seg.pt", image)
        st.image(n_result_image, caption="n-YOLO model Predicted Image", use_column_width=True)
    with colm:
        st.image(f.predict_sample_using_simple_cv(image), caption="opencv-python algorithm predicted Image", use_column_width=True)
    with colr:
        st.image(c.predict_cow(), caption="cow generated Image", use_column_width=True)

st.markdown("""
### Key YOLO Code used!
""")
code_snippet = """
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
"""
st.code(code_snippet, language='python')