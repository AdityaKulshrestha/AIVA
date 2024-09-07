import pyautogui
import time
pyautogui.hotkey('win', 'e')
time.sleep(2)
pyautogui.hotkey('ctrl', 'f')
pyautogui.write('aditya')
pyautogui.press('down')
pyautogui.press('enter')



# Open File Explorer
pyautogui.hotkey('winleft', 'e')
time.sleep(1)

# Type 'Aditya' in the search bar
pyautogui.write('Aditya')
time.sleep(1)

# Press Enter to start the search
pyautogui.press('enter')
