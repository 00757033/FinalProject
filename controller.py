import sys
import os
from ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap , QImage
from PyQt5.QtWidgets import QMessageBox

from pathlib import Path , PurePath
import yaml
import argparse
import cv2
import sys
import json
import shutil 
from PIL import Image
import pandas as pd
import numpy as np
sys.path.append("..")
from yolo.myval import main 
from yolo.mydetect import run
from yolo.iou import get_max_iou

from Unet.mymaskpredict import predict_segment

# from yolov5.iou import get_max_iou
# from Unet.mymaskpredict import predict_segment
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from yolov5.utils.plots import Annotator, colors, save_one_box
# from yolov5.utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow_controller,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.set_value()

    def setup_control(self):
        self.ui.ImgFolderBtn.clicked.connect(self.select_folder)
        self.ui.ImgBtn.clicked.connect(self.select_img)
        self.ui.DetectBtn.clicked.connect(self.detect_click)
        self.ui.SegmentBtn.clicked.connect(self.segment_click)

    def set_value(self):
        self.folderPath = None
        self.imgPath = None
        self.imgCntList = []
        self.imgfolder = "\\images\\"
        self.labelfolder = "\\labels\\"
        self.labelImg = "\\labelimg\\"
        self.maskfolder = "\\masks\\"
        self.mask_name = "_mask.png"
        self.label_name = "_label.png"

        self.className = ['powder_uncover', 'scratch', 'powder_uneven']
        self.h = 0 
        self.w = 0
        self.imgName = None
        self.imgCntList = []
        self.saveYaml =None
        self.ui.FolderPathLabel.setText('No folder loaded')
        self.ui.ImgLabel.setText('No image loaded')
        self.ui.CurrentLabel.setText("Current Image : 0 / 0")

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./") 
        print(folder_path) 
        return  folder_path 

    def load_img(self, path):
        img_path ,_= QFileDialog.getOpenFileName(self,
                  "Open image",
                  path,
                  "Image files (*.jpg *.gif *.png *.jpeg)")
        return img_path

    def select_folder(self):
        self.folderPath = self.load_folder()
        if self.folderPath is None :
            QMessageBox.about(self, "check", "No folder loaded, please confirm")
            return
        if  self.folderPath is None:
            self.ui.FolderPathLabel.setText('No image loaded')
        else : 
            self.ui.FolderPathLabel.setText(self.folderPath)

        self.yaml_file()

        imgsFolderPath = self.folderPath + self.imgfolder

        if Path(imgsFolderPath).is_dir() and Path(imgsFolderPath).exists():
            print("images folder exists")
        else:
            QMessageBox.about(self, "check", "No images folder, please confirm")
            return 

        self.get_mAP()
        self.get_current_list(imgsFolderPath)
        self.ui.CurrentLabel.setText(f"Current Image : 0 / {len(self.imgCntList)}")

        print("Finish load folder")   

    def yaml_file(self):
        d = {'train': self.folderPath+self.imgfolder , 'val':self.folderPath+self.imgfolder , 'names': { 0: 'powder_uncover', 1: 'scratch', 2: 'powder_uneven'}}
        self.saveYaml = self.folderPath + '/output.yaml'
        with open(self.saveYaml, 'w') as f:
            yaml.dump(d, f)
        if self.saveYaml is None :
            QMessageBox.about(self, "check", "No yaml file, please confirm")
            return

     
    def get_mAP(self):
        if self.saveYaml is None :
            self.yaml_file()

        opt = yolo_parse_opt(self.saveYaml)
        result = main(opt)
        if result is None:
            print("No result")
            return
        if len(result) >= 4:
            self.ui.MeanLabel.setText("Folder(Mean) : " + str(result[0]))
            self.ui.UncoverLabel.setText("AP50(uncover): " + str(result[1]))
            self.ui.ScratchLabel.setText("AP50(uneven): " + str(result[2]))
            self.ui.UnevenLabel.setText("AP50(scratch): " + str(result[3]))   
        else:
            print(result)
            QMessageBox.about(self, "check", "No mAP result, please confirm")
            return

    def xywh2xyxy(self, dataset):

        result = []
        for data in dataset:
            # print(data)
            data = data.tolist()
            # print(data)
            if data :
                x = float(data[0]) * self.w
                y = float(data[1]) * self.h 
                w = float(data[2]) * self.w
                h = float(data[3]) * self.h 
            else:
              return None                    
                        
            min_x, min_y = x - (w / 2), y - (h / 2)
            max_x, max_y = x + (w / 2), y + (h / 2)
            result.append([int(min_x), int(min_y), int(max_x), int(max_y)])

        result = np.array(result) 
        return result

    def get_current_list(self,imgsFolderPath):
        # get current image index
        self.imgCntList = []
        for file in  Path(imgsFolderPath).iterdir():
            if file.is_file():
                self.imgCntList.append(file.name)
        print(self.imgCntList)

    def imgs_path(self):
        imgsFolderPath = self.folderPath + self.imgfolder
        self.imgPath = self.load_img(imgsFolderPath)

        if self.imgPath is None :
            QMessageBox.about(self, "check", "No image loaded, please confirm")
            return
        return imgsFolderPath

    def show_img(self, mode, imgPath):
        img = cv2.imread(imgPath)
        img= cv2.resize(img, (280, 280), interpolation=cv2.INTER_AREA)

        if len(img.shape) == 3:
            h, w, c = img.shape
        else:
            h, w= img.shape

        # h, w ,*_ = img.shape

        qImg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        if mode == "original":
            self.ui.OriginalImgBox.setPixmap(QPixmap.fromImage(qImg))
        elif mode == "detect":
            self.ui.DetectionImgBox.setPixmap(QPixmap.fromImage(qImg))
        elif mode == "segment":
            self.ui.SegmentationImgBox.setPixmap(QPixmap.fromImage(qImg))
        elif mode == "detectGT":
            self.ui.DetectionGTImgBox.setPixmap(QPixmap.fromImage(qImg))
        elif mode == "segmentGT":
            self.ui.SegmentationGTImgBox.setPixmap(QPixmap.fromImage(qImg))
        else:
            print("No mode")
            return

    def select_img(self):
        
        if self.folderPath is None :
            QMessageBox.about(self, "check", "No folder loaded, please confirm")
            return

        imgsFolderPath = self.folderPath + self.imgfolder


        if Path(imgsFolderPath).is_dir() and Path(imgsFolderPath).exists():
            self.imgPath = self.load_img(imgsFolderPath)

            if self.imgPath is None :
                QMessageBox.about(self, "check", "No image loaded, please confirm")
                return
            self.imgName = PurePath(self.imgPath).name
            if Path(self.imgPath).exists() :
                self.show_img("original", self.imgPath)
            else:
                QMessageBox.about(self, "check", "No image loaded, please confirm")
                return

            if Path(self.labelImg).exists():
                self.show_img("detectGT", str(str(PurePath(self.imgPath).parents[1]) + self.labelImg + Path(self.imgPath).stem + self.label_name))
            if Path(self.maskfolder).exists():
                self.show_img("segmentGT", str(str(PurePath(self.imgPath).parents[1]) + self.maskfolder + Path(self.imgPath).stem + self.mask_name))


            self.ui.ImgLabel.setText(PurePath(self.imgPath).name)

        else:
            QMessageBox.about(self, "check", "No image folder, please confirm")
            return

        self.get_current_list(imgsFolderPath)

        index = self.imgCntList.index(PurePath(self.imgPath).name) + 1
        self.ui.CurrentLabel.setText(" Current Image : "+str(index) + '/' + str(len(self.imgCntList)))


    def segment_click(self):
        if self.imgPath is None:
            QtWidgets.QMessageBox.about(self, "check", "Please select image")
            return

        gray = cv2.imread(self.imgPath, cv2.IMREAD_GRAYSCALE)
        gray= cv2.resize(gray, (1244, 1254), interpolation=cv2.INTER_AREA)

        grayimgPath = self.folderPath + "/gray/" + PurePath(self.imgName).stem+"_gray.png"

        if not Path(self.folderPath + "/gray/").exists():
            Path(self.folderPath + "/gray/").mkdir(parents=True, exist_ok=True)
            print("create gray folder")

        print("save image in gray levels")
        cv2.imwrite(grayimgPath, gray)


        args = Unet_get_args([grayimgPath])
        predictMaskPath = predict_segment(args)
        print(predictMaskPath)
        self.show_img("segment", predictMaskPath)



        GTPath = self.folderPath + self.maskfolder + PurePath(self.imgName).stem+ self.mask_name

        if Path(GTPath).exists():
            gt = cv2.imread(GTPath, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (1244, 1254), interpolation=cv2.INTER_AREA)
            ret, output1 = cv2.threshold(gt, 25, 255, cv2.THRESH_BINARY)   
            cv2.imwrite(GTPath, output1)

            segimg = cv2.imread(GTPath)   # cimg is a OpenCV image 
            gtArray = np.array(segimg)

        else:
            QtWidgets.QMessageBox.about(self, "check", "Please select image")
            print(Path(GTPath)  )
            return


        if Path(predictMaskPath).exists and Path(GTPath).exists():

            maskimg = cv2.imread(predictMaskPath)   # cimg is a OpenCV image 
            mask_array = np.array(maskimg)

            dice = dice_average_sets(gtArray, mask_array)

            self.ui.DiceCoefficientLabel.setText("Dice Coefficient : " + str(round(dice, 3)))

        else:
            QtWidgets.QMessageBox.about(self, "check", "Please select image")
            print(Path(predictMaskPath) )
            return
                        
    def detect_click(self):
        if self.folderPath is None :
            QMessageBox.about(self, "check", "No folder loaded, please confirm")
            return
        if self.imgPath is None:
            QMessageBox.about(self, "check", "Please select image")
            return

        imgread = cv2.imread(self.imgPath)
        self.h, self.w, _ = imgread.shape

        opt = parse_opt(self.imgPath)

        _, predictFolder, fps = run(**vars(opt))
        self.ui.FPSLabel.setText("FPS : " + str(fps))
                
        predictImgPath = str(predictFolder) +"\\" + self.imgName

    # about bounding box 
        labelPath = self.folderPath + self.labelfolder + PurePath(self.imgName).stem + ".txt"
        predictLabelPath = str(predictFolder) + self.labelfolder + PurePath(self.imgName).stem + ".txt"



        # read xywh from ground ture label
        trueBoxLabel = []
        trueLabel = ""

        if Path(labelPath).exists() and Path(labelPath).is_file():
            with open(labelPath) as f:
                lines = f.readlines()
                for line in lines:

                    line = line.replace("\n","")
                    line = line.split(" ")
                    trueBoxLabel.append(line[1:5])
                    trueLabel = self.className[int(line[0])]
        else:
            QtWidgets.QMessageBox.about(self, "check", "No label file, please confirm")
            print(Path(labelPath))
            return
        self.show_img("detect", predictImgPath)

                
        # read xywh from predict label
        predictBoxLabel = []
        predictLabel = ""
        if Path(predictLabelPath).exists() and Path(predictLabelPath).is_file():
            with open(predictLabelPath) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    predictBoxLabel.append(line[1:5])
                    predictLabel = self.className[int(line[0])]
                    # change string to float 
        else:
            QtWidgets.QMessageBox.about(self, "check", "No label file, please confirm")
            return

        trueBoxLabel = np.array(trueBoxLabel, dtype="f") 
        predictBoxLabel = np.array(predictBoxLabel,dtype="f")
        # change xywh to xyxy for iou compute
        
        trueBoxLabel = self.xywh2xyxy(trueBoxLabel)
        predictBoxLabel = self.xywh2xyxy(predictBoxLabel)
       
        iouList = []

        for t in trueBoxLabel:
            iouList.append(get_max_iou(predictBoxLabel, t))

        if iouList:
            iou = sum(sorted(iouList, reverse = True)[:len(predictBoxLabel)])/len(predictBoxLabel)
        

        self.show_img("detect", predictImgPath)
        self.ui.PredictLabel.setText("Predict : " + predictLabel)
        self.ui.GTLabel.setText("Type(GT): " + trueLabel)
        self.ui.IOULabel.setText("IoU: " + str(round(iou, 3)))

def Unet_get_args(img_list):
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='Unet/checkpoint_epoch250.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', metavar='INPUT', default=img_list, help='Filenames of input images')
    parser.add_argument('--output', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.1,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.05,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()

def dice_average_sets(y_true, y_pred):
    assert y_true.shape[0] == y_pred.shape[0], "you should use same size data"
    dice = []
    for i in range(y_true.shape[0]):
        dice.append(dice_coef(y_true[i], y_pred[i]))
    return np.mean(dice)


def yolo_parse_opt(yaml_file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=yaml_file, help='dataset.yaml path')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo/yolo.pt', help='model path(s)')

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()

    return opt

def dice_coef(y_true:np.ndarray, y_pred:np.ndarray):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    return round((2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth),3)

def parse_opt(path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo/yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=path, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name') 
    parser.add_argument('--line-thickness', default=5, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')

    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    opt = parser.parse_args()
    return opt   

if __name__ =='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())


