import cv2
import pathlib
from PIL import Image
choosefile = ".\\test\\"
images = "images\\"
labels = "labels\\"
labelimgdst = "\\dst\\"
pathlib.Path(choosefile+labelimgdst).resolve().mkdir(parents=True, exist_ok=True)

i = 0 
for file in pathlib.Path(str(choosefile + images)).iterdir(): #走訪某資料夾內的所有檔案與資料夾
    if i > 4:
        break
    i=i+1
    with pathlib.Path(".\\"+str(file).replace("images", "labels").replace(".png", ".txt")).open() as f:
        line = f.readline().split(' ')
        width, height = Image.open(file).size
        
        img = cv2.imread(str(file))

        print(width *float(line[1]),height*float(line[2]),width *float(float(line[1])+float(line[3])),height*(float(line[2])+float(line[4])) )
        cv2.rectangle(img, (int(width *float(line[1])), int(height*float(line[2]))), (int(3*width *float(float(line[1])+float(line[3]))), int(3*height*(float(line[2])+float(line[4])))), (0, 0, 255), -1)
        # # 按下任意鍵則關閉所有視窗
        image = cv2.resize(img, (1254, 1244))
        cv2.imshow('My Image', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()



