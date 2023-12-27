import cv2
from pathlib import Path
# method 1

image = Path('..\\class_data\\Train').resolve()
imagedest = Path('..\\Pytorch-UNet\\data\\imgs').resolve()
renamedest = Path('..\\Pytorch-UNet\\data\\masks_rename').resolve()
Path(imagedest).mkdir(parents=True, exist_ok=True)
Path(imagedest).mkdir(parents=True, exist_ok=True)

mask = Path('..\\class_data\\Train').resolve()
maskdest = Path('..\\Pytorch-UNet\\data\\masks').resolve()
Path(maskdest).mkdir(parents=True, exist_ok=True)

class_names = ['powder_uncover', 'powder_uneven', 'scratch']
types = ['image', 'mask']
# create a new folder to store the converted images
for class_name in class_names:
    for type in types:
        image = Path('..\\class_data\\Train\\'+str(class_name)+'\\'+str(type)).resolve()    
        if type == 'image':
            imagedest = Path('..\\Pytorch-UNet\\data\\imgs\\').resolve()        
        else:
            imagedest = Path('..\\Pytorch-UNet\\data\\masks\\').resolve()

        for img in image.glob('*.png'):
            name = str(img).split('\\')[-1]
            image = cv2.imread(str(img))
            image = cv2.resize(image, (1254, 1244))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if type == 'mask':
                ret , image =  cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)

            # print("str(img)",str(img),'->',str(imagedest)+'\\'+str(name))
            cv2.imwrite(str(imagedest)+'\\'+str(name), image)
            name = str(img).split('.')[0]
   
            cv2.imwrite(str(renamedest)+'\\'+str(name)+'_mask.png', image)

