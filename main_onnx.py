# Import necessary libraries and modules
import onnxruntime as ort
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
from utils.general import (cv2, non_max_suppression, xyxy2xywh)
import torch
from models.common import DetectMultiBackend  # Add this import for YOLO model

# Add these imports for YOLO model
from config import aaMovementAmp, fovCircle, fovCircleSize, useMask, maskHeight, maskWidth, aaQuitKey, confidence, headshotMode, cpsDisplay, visuals, onnxChoice, centerOfScreen
import gameSelection
import pygetwindow as gw
import ctypes
import cupy as cp
# Define a function to set the window on top
def set_window_on_top(window):
    hwnd = ctypes.windll.user32.FindWindowW(None, window.title)
    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)
    ctypes.windll.user32.ShowWindow(hwnd, 1)
    ctypes.windll.user32.SetForegroundWindow(hwnd)

# Define a function to check if the Shift Lock key is enabled
def is_shift_lock_enabled():
    return win32api.GetKeyState(win32con.VK_CAPITAL) & 1 == 1

# Define the main function
def main():
    # Call the gameSelection function to get camera and screen dimensions
    camera, cWidth, cHeight = gameSelection.gameSelection()
    
    # Initialize variables for the main loop
    count = 0
    sTime = time.time()
    
    # Choose the correct ONNX Provider based on the configuration
    if onnxChoice == 1:
        onnxProvider = "CPUExecutionProvider"
    elif onnxChoice == 2:
        onnxProvider = "DmlExecutionProvider"
    elif onnxChoice == 3:
        onnxProvider = "CUDAExecutionProvider"

    # Configure ONNX session options
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Initialize ONNX Inference Session with GPU provider
    ort_sess = ort.InferenceSession('yolov5s320Half.onnx', sess_options=so, providers=['CUDAExecutionProvider'])

    # Generate random colors for bounding boxes
    COLORS = np.random.uniform(0, 255, size=(1500, 3))
    last_mid_coord = None

    # Main loop: exit when 'aaQuitKey' is pressed
    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
        # Check if Shift Lock is enabled
        if is_shift_lock_enabled():
            # Get the latest frame from the camera
            npImg = np.array(camera.get_latest_frame())

            # Apply mask to the frame if 'useMask' is enabled
            if useMask:
                npImg = apply_mask(npImg, targets)

            # Prepare the input image for ONNX model
            if onnxChoice == 3:
                im = torch.from_numpy(npImg).to('cuda')
                if im.shape[2] == 4:
                    im = im[:, :, :3,]

                im = torch.movedim(im, 2, 0)
                im = im.half()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]
            else:
                im = np.array([npImg])
                if im.shape[3] == 4:
                    im = im[:, :, :, :3]
                im = im / 255
                im = im.astype(np.half)
                im = np.moveaxis(im, 3, 1)

            # Run the ONNX model and get predictions
            outputs = ort_sess.run(None, {'images': cp.asnumpy(im)})

            im = torch.from_numpy(outputs[0]).to('cpu')

            # Perform non-maximum suppression to get the final bounding boxes
            pred = non_max_suppression(
                im, confidence, confidence, 0, False, max_det=10)

            # Process the detected targets
            targets = []
            for i, det in enumerate(pred):
                s = ""
                gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
                if len(det):
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {int(c)}, "

                    for *xyxy, conf, cls in reversed(det):
                        targets.append((xyxy2xywh(torch.tensor(xyxy).view(
                            1, 4)) / gn).view(-1).tolist() + [float(conf)])

            # Create a DataFrame with target information
            targets = pd.DataFrame(
                targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

            # Calculate the center of the screen
            center_screen = [cWidth, cHeight]

            # Process targets if there are any
            if len(targets) > 0:
                # Sort targets based on distance from the center
                if centerOfScreen:
                    targets["dist_from_center"] = np.sqrt((targets.current_mid_x - center_screen[0])**2 + (targets.current_mid_y - center_screen[1])**2)
                    targets = targets.sort_values("dist_from_center")

                # Sort targets based on distance from the last detected target
                if last_mid_coord:
                    targets['last_mid_x'] = last_mid_coord[0]
                    targets['last_mid_y'] = last_mid_coord[1]
                    targets['dist'] = np.linalg.norm(
                        targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                    targets.sort_values(by="dist", ascending=False)

                # Get the coordinates and dimensions of the first target
                xMid = targets.iloc[0].current_mid_x
                yMid = targets.iloc[0].current_mid_y
                box_height = targets.iloc[0].height

                # Adjust the mouse movement based on headshot mode
                if headshot_mode:
                    headshot_offset = box_height * 0.37
                else:
                    headshot_offset = box_height * 0.2

                # Calculate the mouse movement based on the target position
                mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]

                # Calculate the distance from the center of the screen
                dist_from_center = np.sqrt(mouseMove[0]**2 + mouseMove[1]**2)

                # Move the mouse if the target is within the FOV circle
                if fovCircle:
                    if dist_from_center <= fovCircleSize:
                        if win32api.GetKeyState(0x02) < 0:
                            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp), 0, 0)

                # Move the mouse without checking FOV circle
                else:
                    if win32api.GetKeyState(0x02) < 0:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(
                            mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp), 0, 0)
                    last_mid_coord = [xMid, yMid]

            # Triggerbot Alt for Toggle
            if len(targets) > 0 and win32api.GetKeyState(0xA4) and abs(mouseMove[0]) <= 50 and abs(mouseMove[1]) <= 50:
                # Press the mouse button
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                # Release the mouse button
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

            # Visualize targets if 'visuals' is True
            if len(targets) > 0 and visuals:  # Check if targets are detected and visuals is True
                npImg = cp.asnumpy(npImg[0])

                # Display FOV Circle
                if fovCircle:
                    cv2.circle(npImg, (cWidth, cHeight), fovCircleSize, (0, 0, 255), 2)

                # Loop over every item identified and draw a bounding box
                for i in range(0, len(targets)):
                    halfW = round(targets["width"][i] / 2)
                    halfH = round(targets["height"][i] / 2)
                    midX = targets['current_mid_x'][i]
                    midY = targets['current_mid_y'][i]
                    (startX, startY, endX, endY) = int(
                        midX + halfW), int(midY + halfH), int(midX - halfW), int(midY - halfH)

                    idx = 0
                    # Draw the bounding box and label on the frame
                    label = "{}: {:.2f}%".format(
                        "Human", targets["confidence"][i] * 100)
                    cv2.rectangle(npImg, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(npImg, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # Forced garbage cleanup every second
            count += 1
            if (time.time() - sTime) > 1:
                if cpsDisplay:
                    print("CPS: {}".format(count))
                count = 0
                sTime = time.time()

            # Visualize what the Aimbot sees
            if len(targets) > 0 and visuals:
                # Check if the window exists
                if not any(window.title == 'Live Feed' for window in gw.getWindowsWithTitle('Live Feed')):
                    cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Live Feed', cWidth, cHeight)
                cv2.imshow('Live Feed', npImg)
                live_feed_window = gw.getWindowsWithTitle('Live Feed')[0]
                set_window_on_top(live_feed_window)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    exit()

# Function to apply the mask to the frame
def apply_mask(frame, targets):
    for i in range(len(targets)):
        halfW = round(targets["width"][i] / 2)
        halfH = round(targets["height"][i] / 2)
        midX = int(targets['current_mid_x'][i])
        midY = int(targets['current_mid_y'][i])
        startX, startY, endX, endY = (
            midX - halfW, midY - halfH, midX + halfW, midY + halfH)

        # Mask the detected target
        frame[startY:endY, startX:endX, :] = 0

    return frame

# Entry point of the script
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Handle exceptions and print error information
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        print(str(e))
        print("Ask @Wonder for help in our Discord in the #ai-aimbot channel ONLY: https://discord.gg/rootkitorg")
