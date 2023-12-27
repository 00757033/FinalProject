import json
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import shutil
import pathlib # for creating directory
import argparse
from pathlib import PurePath

#
newdir = "..\\newdata\\"
dirdst = ["images", "labels", "labelimg", "masks"]

mask_name = "_mask.png"
label_name = "_label.png"

pathlib.Path(newdir + dirdst[0]+ "\\").mkdir(parents=True, exist_ok=True)
pathlib.Path(newdir + dirdst[1]+ "\\").mkdir(parents=True, exist_ok=True)
pathlib.Path(newdir + dirdst[2]+ "\\").mkdir(parents=True, exist_ok=True)
pathlib.Path(newdir + dirdst[3]+ "\\").mkdir(parents=True, exist_ok=True)
original_img_path = '..\\sample'

class_names = ['powder_uncover', 'powder_uneven', 'scratch']
yolo_csv = pd.DataFrame(
    columns=["category", "x", "y", "w", "h", "image_name", "set_type", "image_path"])

# convert the label to yolo format
def yolo_format(convert_img_file: str, save_img_file_name: str):

    # get label path
    label_path = convert_img_file.replace("image", "label").replace(".png", ".json")

    # get image name
    image_name = str(PurePath(convert_img_file).stem)

    # read the label file
    with open(label_path, "r") as file:
        json_file = json.load(file)

    # get the image size
    width, height = Image.open(convert_img_file).size

    # process the label file
    for annotation in json_file["shapes"]:
        if annotation["label"] == "powder_uncover":
            category_id = 0
        elif annotation["label"] == "powder_uneven":
            category_id = 1
        else:
            category_id = 2
        points = annotation["points"]

    # process the label file

    with open( newdir + dirdst[1]+ "\\"+image_name+".txt" , "w") as file:
        for annotation in json_file["shapes"]:

            x_min, y_min = points[0]
            x_max, y_max = points[1]

            # convert the bounding box to yolo format
            x = (x_min + (x_max-x_min)/2) * 1.0 / width
            y = (y_min + (y_max-y_min)/2) * 1.0 / height
            w = (x_max-x_min) * 1.0 / width

            # the height have nagative value, so we need to use abs() to get the positive value
            h = abs((y_max-y_min) * 1.0 / height)

            yolo_format_data = str(category_id) + " " + str(x) + \
                " " + str(y) + " " + str(w) + " " + str(h)

            yolo_csv.loc[len(yolo_csv)] = [category_id, x, y, w, h,
                                        image_name, save_img_file_name, convert_img_file]
        
            file.write(yolo_format_data)
            file.write("\n")


if __name__ == "__main__":
    for class_name in class_names:
        class_dir = os.path.join(original_img_path, class_name)

        # concatenate the image, label and mask directories
        for filename in tqdm(os.listdir(class_dir)):

            # check the file type
            file_type = str(PurePath(filename).suffix)
            file_name = str(PurePath(filename).stem)

            if file_type ==".png":
                if file_name+".json" in os.listdir(class_dir):
                    yolo_format(convert_img_file=f"{class_dir}\\{filename}", save_img_file_name='sample')
                    
                    shutil.copy(class_dir+"\\"+ filename, newdir + dirdst[0]+ "\\" + filename)                
                    shutil.copy(class_dir+"\\"+file_name+mask_name,newdir + dirdst[3] + "\\" +file_name+mask_name)
                    shutil.copy(class_dir+"\\"+file_name+label_name,newdir + dirdst[2]+ "\\" +file_name+label_name)
