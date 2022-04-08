import pytesseract
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import date
import json

load_dotenv()
loc_id = os.getenv('LOCATION_ID')

root_dir = str(loc_id)
today = date.today()
d1 = today.strftime("%d%m%Y")

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
pytesseract.pytesseract.tesseract_cmd = r'D:\github\py_tess\exe\Tesseract-OCR\tesseract.exe'

# def extract_text(dir_path):

#     for root, directories, files in os.walk(dir_path):
#         for file in files:
#             print(file)

def show_image(img, **kwargs):
    """
    Show an RGB numpy array of an image without any interpolation
    """
    plt.subplot()
    plt.axis('off')
    plt.imshow(
        X=img,
        interpolation='none',
        **kwargs
    )

def apply_morphology(img, method):
    """
    Apply a morphological operation, either opening (i.e. erosion followed by dilation) or closing (i.e. dilation followed by erosion). Show result.
    """
    if method == 'open':
        op = cv2.MORPH_OPEN
    elif method == 'close':
        op = cv2.MORPH_CLOSE

    img_morphology = cv2.morphologyEx(
        src=img,
        op=op,
        kernel=np.ones((5, 5), np.uint8),
    )

    cv2.imshow('mor', img_morphology)

    return img_morphology


def get_lpno(root_dir):
    # for root, directories, files in os.walk(root_dir):
    #     for directory in directories:
    #         if directory == d1:

    #         print(directory)
    #     for file in files:
    #         print(file)

    lpno_ocr_dict = {}
    
    for dirName, subdirList, fileList in os.walk(root_dir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if fname.endswith('crop.jpg'):
                vehicle_id = fname.split('_')[0]
                crop_path = os.path.join(dirName, fname)
                print('\t%s' % fname)
                print(f'Final path: {crop_path}')

                gray = cv2.imread(crop_path, 0)
                gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
                blur = cv2.GaussianBlur(gray, (1,1), 0)
                gray = cv2.medianBlur(gray, 1)
                # perform otsu thresh (using binary inverse since opencv contours work better with white text)
                ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
                # cv2.imshow("Otsu", thresh)
                # cv2.waitKey(0)
                rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

                # apply dilation 
                dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
                #cv2.imshow("dilation", dilation)
                #cv2.waitKey(0)
                # find contours

                # dilation = apply_morphology(dilation, 'close')

                try:
                    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

                # create copy of image
                im2 = gray.copy()


                plate_num = ""
                # loop through contours and find letters in license plate
                for cnt in sorted_contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    height, width = im2.shape
                    
                    # if height of box is not a quarter of total height then skip
                    # if height / float(h) > 15: continue
                    ratio = h / float(w)
                    # if height to width ratio is less than 1.5 skip
                    # if ratio < 1.5: continue
                    area = h * w
                    # if width is not more than 25 pixels skip
                    if width / float(w) > 15: continue
                    # if area is less than 100 pixels skip
                    if area < 100: continue
                    # draw the rectangle
                    rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
                    roi = thresh[y-5:y+h+5, x-5:x+w+5]
                    roi = cv2.bitwise_not(roi)
                    roi = cv2.medianBlur(roi, 5)
                    #cv2.imshow("ROI", roi)
                    #cv2.waitKey(0)
                    text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                    #print(text)
                    plate_num += text
                lpno_ocr_dict[vehicle_id] = plate_num.split('\n')[0]
                return lpno_ocr_dict

lpnos = get_lpno(root_dir)
print("LP NOs: ",lpnos)

json_object = json.dumps(lpnos, indent = 4)
  
# Writing to sample.json
with open("lp_data.json", "w") as outfile:
    outfile.write(json_object)

# print('Text: ',plate_num)

# cv2.imshow("Character's Segmented", im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

