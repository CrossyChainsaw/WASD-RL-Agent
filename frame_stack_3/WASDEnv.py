import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import ImageGrab, Image
import pyautogui
import win32gui
import cv2
import win32con
from torchvision import transforms

class WinFormsGameEnv(gym.Env):
    def __init__(self):
        super(WinFormsGameEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 possible actions: W, A, S, D
        # Define the observation space: grayscale full-screen screenshot
        screen_width, screen_height = pyautogui.size()  # Get full screen dimensions
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 1), dtype=np.uint8)
        self.window_title = "WASD" # game name
        self.window_handle = find_window(window_title=self.window_title)

    def step(self, action):
        do_action(action=action)                    # do action
        observation = get_state(self.window_handle) # new state
        reward = self.calculate_reward()            # check reward
        done = self.is_game_over(reward)            # check game over
        return observation, reward, done, False, {} # return

    def reset(self, seed=None, options=None):
        # Reset the game (you need to implement this part)
        #self.reset_game()
        # Capture the initial screenshot as the initial observation (convert to grayscale)
        observation = get_state(self.window_handle)
        return observation, {}

    def render(self):
        # Capture the initial screenshot as the initial observation (convert to grayscale)
        observation = get_state(self.window_handle)
        return observation

    def calculate_reward(self):
        score_img = get_score_img()
        accuracy = check_score(score_img)
        if accuracy > 0.9:
            return 0
        else:
            return 1    

    def is_game_over(self, reward):
        if reward == 0:
            return True
        else:
            return False

    def reset_game(self):
        state = get_state(self.window_handle)
        return state

def find_window(window_title:str):
    # Get the handle of the window
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        raise Exception(f"Window with title '{window_title}' not found.")
    return hwnd

def get_state(window_handle):
    #print(f"Window handle: {window_handle}") # sometimes you might need this line to fix weird win32gui.SetForegroundWindow(window_handle) bug
    win32gui.SetForegroundWindow(window_handle)
    screenshot = ImageGrab.grab() # capture full screen
    screenshot_np = np.array(screenshot)
    screenshot_np = preprocess_frame(screenshot_np)
    return screenshot_np

def get_score_img():
    score_screenshot = ImageGrab.grab(bbox=(440, 50, 510, 160)) # capture the score number
    score_screenshot_np = np.array(score_screenshot)
    return score_screenshot_np

def do_action(action:int):
    # Simulate key press for the action
    if action == 0:
        pyautogui.press('w')
    elif action == 1:
        pyautogui.press('a')
    elif action == 2:
        pyautogui.press('s')
    elif action == 3:
        pyautogui.press('d')

def check_score(image, template_path='0_temp.png'):
    # Load template (image of "0")
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # Read template as grayscale

    # Convert the game screen to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Match template
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

    # Get the best match position and value
    _, max_val, _, _ = cv2.minMaxLoc(result)

    return max_val

# Preprocess the frames (resize and convert to grayscale)
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((84, 84)),  # Resize to 84x84
        transforms.ToTensor(),  # Convert to tensor, will have shape [1, 84, 84]
    ])
    return transform(frame).squeeze(0)  # Remove the single channel dimension, resulting in [84, 84]