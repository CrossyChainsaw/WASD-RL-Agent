from PIL import Image
import cv2
import numpy as np
import time
import pyautogui
from torchvision import transforms

def alt_tab():
    pyautogui.keyDown('alt')
    time.sleep(.2)
    pyautogui.press('tab')
    time.sleep(.2)
    pyautogui.keyUp('alt')

def tensor_to_pil(tensor):
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)