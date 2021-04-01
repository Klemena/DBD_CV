"""
Refs
https://stackoverflow.com/questions/33863921/detecting-a-keypress-in-python-while-in-the-background
https://stackoverflow.com/questions/53551676/python-screenshot-of-background-inactive-window
"""

"""
General TODO's
make delim images more modular in size, Importance: HIGH
open_gui(), Importance: MED
segment_img(), Importance: Low

"""
from ctypes import windll

import cv2
import win32gui
import win32ui
import numpy as np
from PIL import Image, ImageFilter
import glob
from pyWinhook import HookManager
import pytesseract
import matplotlib.pyplot as plt
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

def segment_img(img):
    pass

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    text = pytesseract.image_to_string(img)  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text = (text.partition("\n")[0])
    print(text)
    return text


def find_subimg(img1, img2,matching_threshold = 0.7, vert_threshold = 5,resize=False):
    """
    :param img1: Larger image (Screenshot)
    :param img2: Smaller image (Icon)
    :param matching_threshold: A normalized threshold value to establish matching
    :param vert_threshold: A threshold value in pixels to determine minimum distance between features
    :param resize: used for debugging. Resizing an image - to be implimented legit later
    :return: list of coordinates of instances of img2 within img1
    """

    img_rgb = cv2.imread(img1)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    """
    im = cv2.imread("iconx.png", -1)

# extract the alpha channel:
a:= im[:,:,3]

#use that as a mask for bitwise compositing:
im2 = cv2.bitwise_and(im,im,mask=a)

#convert *that* to grayscale
im2 = cv2.cvtColor(im2,cv2.COLOR_BGRA2GRAY)
    """

    template = cv2.imread(img2)

    if resize:
        template = cv2.imread(img2,-1)
        #TODO put in a conditional that slice must agree in size a:=...
        a = template[:, :, 3]
        template = cv2.bitwise_and(template, template, mask=a)
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
        template = cv2.resize(template,(47,47))

    else:
        template = cv2.imread(img2,0)


    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= matching_threshold)
    regions = []
    unique_y = []

    for pt in zip(*loc[::-1]):
        is_unique = True
        for unique_pt in unique_y:
            if (abs(unique_pt - pt[1])) < vert_threshold:
                is_unique = False
        if is_unique:
                unique_y.append(pt[1])
                current_region = [pt, (pt[0] + w, pt[1] + h)]
                regions.append(current_region)
                #Print and highlight region
                #print(current_region)
                #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    #Display highlighted image
    #cv2.imshow('Detected', img_rgb)

    return regions

def crop_LR(img_file, features):
    """
    :param img: Original image (Screenshot)
    :param features: A list of coordinates of features that should be contained in the result.
    :return: a cropped image to contain the features at edges (removes excess)
    """
    img = Image.open(img_file)
    min_x, max_y= img.size
    max_x = 0
    min_y = 0
    for feature in features:
        for point in feature:
            min_x = min(min_x,point[0])
            max_x = max(max_x,point[0])

    img = img.crop((min_x,min_y,max_x,max_y))
    #img.show()
    return img

def crop_to_each_player(img,player_coordinates):
    """
    :param img: Source image
    :param features: A list of coordinates of delimiter regions
    :return: An array of images for each player
    """
    # Sort the slices in ascending order
    y_coords = []
    for pair in player_coordinates:
        y_coords.append(pair[1][1])
    y_coords.sort()

    players = []
    max_x, max_y= img.size

    # For each player_coordinate - 2 (top and bottom don't count) copy the image and crop it between each 2 points
    #TODO add assert to ensure size
    for i in range(len(player_coordinates) - 1):
        players.append(img)
        players[i] = players[i].crop((0,y_coords[i],max_x,y_coords[i+1]))
        players[i] = players[i].convert("L")
        #players[i] = players[i].filter(ImageFilter.FIND_EDGES)

    return players

def basic_perk_search(player):
    w,h = player.size
    width_crop_ratio_min = int(w/5) #Found through testing
    width_crop_ratio_max = int(w/2.19) #Found through testing
    cropper_player = player.crop((width_crop_ratio_min, 0, width_crop_ratio_max, h))
    cropper_player.save("temp.png", "PNG")
    cropper_player.show()
    ocr_core("temp.png")
    new_thresh = 0.62 #Found through testing
    guesses = []
    #Foothold, next step is going to be going into \Perks, for each file in it -> guesses.append((${NAME},len(loc))
    for filename in os.listdir("Perks"):
        query = "Perks\\"+filename
        loc = find_subimg("temp.png", query, new_thresh,resize = True)
        query = query.replace("Perks\iconPerks_","")
        query = query.replace(".png","")
        pair = (query,str(len(loc)))
        guesses.append(pair)

    for guess in guesses:
        if int(guess[1]) > 0:
            print("Perk: " + guess[0] +"\t isPresent: " + guess[1])

def img_analyze_tests():
    #TODO all img scalefactor (guess is something like 256/x = 47/1920)
    img = "image_2.png"
    img_delim = "delimiter.png"
    img_status = "status.png"
    img_fail = "template_fail.png"
    img_success = "template_success.png"
    #print("Test 1")
    player_delims = find_subimg(img, img_delim)
    #print("Player delims: " + str(player_delims))
    #TODO add assert statments to assert length 4
    feedback_delim = find_subimg(img, img_status)
    #print("Feedback delims: " + str(feedback_delim))
    #TODO add assert statments to assert length 1
    cropped_img = crop_LR(img,player_delims+feedback_delim)
    player_imgs = crop_to_each_player(cropped_img,player_delims+feedback_delim)
    for player in player_imgs:
        basic_perk_search(player)
        print("\n")

    #print("Test 2")
    #find_subimg(img, img_delim_2)
    #segment_img(img)
    #ocr_core(img)
    """
    print("Test 2")
    subimg(img,img_success)
    print("Test 3")
    subimg(img, img_fail)
    
    """
def winEnumHandler( hwnd, ctx ):
    if win32gui.IsWindowVisible( hwnd ):
        print (hex(hwnd), win32gui.GetWindowText( hwnd ))

def get_bitmap(window_name,output_name):
    """Get and return a bitmap of the Window."""
    window_handle = win32gui.FindWindow(None, window_name)
    if window_handle == 0:
        print("Dead By Daylight is currently closed...")
        return
    left, top, right, bot = win32gui.GetWindowRect(window_handle)
    w = right - left
    h = bot - top
    #print("BP 1")
    hwnd_dc = win32gui.GetWindowDC(window_handle)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(save_bitmap)
    windll.user32.PrintWindow(window_handle, save_dc.GetSafeHdc(), 0)
    bmpinfo = save_bitmap.GetInfo()
    bmpstr = save_bitmap.GetBitmapBits(True)
    # This creates an Image object from Pillow
    bmp = Image.frombuffer('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr, 'raw', 'BGRX', 0, 1)
    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(window_handle, hwnd_dc)
    output_name = output_name + ".png"
    bmp.save(output_name)

class Keystroke_Watcher(object):
    def __init__(self):
        self.hm = HookManager()
        self.hm.KeyDown = self.on_keyboard_event
        self.hm.HookKeyboard()

    def on_keyboard_event(self, event):
        #print('KeyID:', event.KeyID) #This will display the key ID for all keyboard inputs
        try:
            if event.KeyID  == 103:
                self.take_DBD_screenshot()
            if event.KeyID == 104:
                win32gui.EnumWindows(winEnumHandler, None)
        finally:
            return True

    def take_DBD_screenshot(self):
        window_name = "DeadByDaylight  "
        print('Taking DBD Screenshot...')
        output_name = "DBD_" + str(len(glob.glob('DBD_*')) + 1)
        get_bitmap(window_name,output_name)

    def shutdown(self):
        win32gui.PostQuitMessage(0)
        self.hm.UnhookKeyboard()

def key_watcher():
    watcher = Keystroke_Watcher()
    win32gui.PumpMessages()

def open_gui():
    pass

if __name__ == "__main__":
    open_gui()
    # img_analyze_tests()
    #key_watcher()