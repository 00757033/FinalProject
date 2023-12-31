import cv2

# 設定輸入與輸出
inputLabelPath = 'runs\detect\exp22\labels\powder_uncover_0139.txt'
# inputLabelPath = r"C:\Users\Xsheep\Desktop\yolov5-custom\data\YOLODataset\labels\train\powder_uncover_0139.txt"
inputImgPath = r'C:\Users\Xsheep\Desktop\yolov5-custom\runs\detect\exp22\powder_uncover_0139mydraw.png'
outputImgPath = r'C:\Users\Xsheep\Desktop\yolov5-custom\runs\detect\exp22\powder_uncover_0139mydraw2.png'

# 載入圖片
cv2image = cv2.imread(inputImgPath)
print(cv2image.shape)
img_w, img_h, c = cv2image.shape

with open(inputLabelPath) as f:
  for line in f:
    line = line.strip() # 刪除多餘的空白
    data = line.split() # 將 YOLO 一列的內容轉成一維陣列
    # 將 YOLO 格式轉換為邊界框的左上角和右下角座標
    bbox_width = float(data[3]) * img_w
    bbox_height = float(data[4]) * img_h
    center_x = float(data[1]) * img_w 
    center_y = float(data[2]) * img_h 
    
        
    min_x, min_y = center_x - (bbox_width / 2), center_y - (bbox_height / 2)
    max_x, max_y = center_x + (bbox_width / 2), center_y + (bbox_height / 2)
    print(min_x,min_y,max_x,max_y)
    # 在圖片上寫上物件名稱
    # cv2.putText(cv2image, 'Altolamprologus compressiceps', (int(min_x), int(min_y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # 畫出邊界框
    cv2.rectangle(cv2image, (int(min_x),int(min_y)), (int(max_x),int(max_y)), (255,255,255), 2)
    # 將檔案另存新圖片檔
    cv2.imwrite(r'{}'.format(outputImgPath), cv2image)
  f.close()