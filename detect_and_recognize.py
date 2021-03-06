# %%
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from four_points_transform import four_points_transform
import pytesseract as pt




def get_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    blured = cv2.bilateralFilter(gray, 11, 17, 17)
    ret,th1 = cv2.threshold(blured,150,255,cv2.THRESH_BINARY)
    edged = cv2.Canny(th1, 100, 200)

    dilated = cv2.dilate(edged, (3,3))

    keypoints = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    cnts = imutils.grab_contours(keypoints)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]


    location = None
    cropped_image = None
    points = None
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        # print(peri)
        approx = cv2.approxPolyDP(cnt, 0.05*peri , True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 3000:
            location = approx
            break
    
    if location is None: 
        print("No plate detected!")
        cv2.rectangle(image, (0,0), (165,35), (255,255,255), -1)
        cv2.putText(image, 'Can\'t detect plate!', (2,25), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        cv2.imshow('result', image)

    elif location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1, )
        new_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('localtion', new_image)

        v1 = location[0][0]
        v2 = location[1][0]
        v3 = location[2][0]
        v4 = location[3][0]
        points =  np.array([v1, v2, v3, v4])
        cropped_image, order_points = four_points_transform(gray, points) # get bird-eye view
        cv2.imshow('croped-image', cv2.resize(cropped_image, (0,0), fx=3, fy=3))
        
        ord_points = [list(map(int, point)) for point in order_points]
        # print(order_points)
    return cropped_image, ord_points


def binarize(plate_img):
    gray = cv2.resize( plate_img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.medianBlur(gray, 3)
    
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    cv2.imshow('threshold', thresh)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0] if len(cnts) == 2 else cnts[1]
    mask2 = np.zeros(thresh.shape, np.uint8)
    cv2.fillPoly(mask2, cnt, [255,255,255])
    mask2 = 255 - mask2
    result = cv2.bitwise_or(thresh, mask2)

    cv2.imshow('binary image', result)
    return thresh

def read_plate(binary_img):
    high_part =binary_img[:][:int(binary_img.shape[1]/2.2)]
    low_part = binary_img[:][int(binary_img.shape[1]/2.5):]

    pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    high = pt.image_to_string(high_part, lang ='eng',config ='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text_high = "".join(high.split()).replace(":", "").replace("-", "")
    text_low = pt.image_to_string(low_part,  config='-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3')
    text = text_high+' '+text_low[:-1]
    print(text)
    return text


##############################################



def main(path):
    # path = str(input('Your image path: '))
    image = cv2.imread(path)
    cv2.imshow('original', image)

    (cropped_plate, points) = get_plate(image)
    if cropped_plate is not None:
        binary_plate = binarize(cropped_plate)
        text = read_plate(binary_plate)

        cv2.line(image, points[0], points[1], (0,255,0), 3)
        cv2.line(image, points[1], points[2], (0,255,0), 3)
        cv2.line(image, points[2], points[3], (0,255,0), 3)
        cv2.line(image, points[3], points[0], (0,255,0), 3)
        cv2.rectangle(image, (0,0), (160,35), (255,255,255), -1)
        cv2.putText(image, text, (10,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)
        cv2.imshow('result', image)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()

path = r'test_images\0000_00532_b.jpg'
main(path)

