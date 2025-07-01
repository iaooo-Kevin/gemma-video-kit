"""
This script utility functions for Gemma Vision to leverage the video capabilities of Gemma.
However, it is implored that you use this locally
"""
import PIL.Image
import numpy as np
import cv2
from typing import List, Dict, Optional, Union, Tuple
import os
from PIL import Image
import tempfile
import PIL
import warnings
import requests #If it is a URL.

def downSample(video: str, N: int = 16) -> Tuple[List[PIL.Image.Image], List[float]]:
    """
    Downsamples a video to N frames.
    Args:
        video (str): Path to the video file or URL.
        N (int): Number of frames to sample from the video. Defaults to 16.
    Returns:
        List[PIL.Image.Image]: List of PIL Image objects representing the sampled frames.
    """
    if "http" in video:
        response = requests.get(video)
        #Create a temporary file to save the downloaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tempFile:
            tempFile.write(response.content)
            tempFile.flush() #Stores the file in memory.
            video = tempFile.name
        capture = cv2.VideoCapture(video)
    else:
        capture = cv2.VideoCapture(video)
    
    FPS = capture.get(cv2.CAP_PROP_FPS)
    totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    indices = np.linspace(0, totalFrames - 1, N, dtype=int)
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = capture.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            timestamp = round(index / FPS, 2)
            frames.append((image, timestamp))
    capture.release()
    return frames

def constructInputMessage(frames: Tuple[List[PIL.Image.Image], List[float]], Prompt: str, SysInstructions: str, tempPath: str) -> List[Dict[str, Union[str, PIL.Image.Image]]]:
    """
    Constructs the input message for Gemma Vision from the sampled frames.
    Args:
        frames (Tuple[List[PIL.Image.Image], List[float]]): Tuple containing a list of PIL Image objects and their timestamps.
        Prompt (str): The prompt accompanying the video.
        SysInstructions (str): Instructions for the Gemma Vision model. Its raison d'être, so to speak. If None is provided, a default will be used.
        tempPath (str): Temporary path to save the images. (Required)
    Returns:
        List[Dict[str, Union[str, PIL.Image.Image]]]: List of dictionaries with 'image' and 'timestamp' keys.
    """
    if not os.path.exists(tempPath):
        os.makedirs(tempPath)
    SysInstructions = SysInstructions or "You are a helpful assistant."
    if Prompt is None:
        warnings.warn("No prompt provided. Using default prompt: 'Summarise the video content.'")
        Prompt = "Summarise the video content."
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SysInstructions}]
        },
        {
            "role": "user",
            "content": []
        }
    ]

    for data in frames:
        image, timestamp = data
        messages[1]['content'].append({
            "type": "text",
            "text": f"Frame at {timestamp} seconds."
        })
        image.save(tempPath+f"/frame_{timestamp}.png")
        messages[1]['content'].append({
            "type": "image",
            "url": f"{tempPath}/frame_{timestamp}.png"
        })
    return messages

if __name__ == "__main__":
    #Example usage.
    videopath = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
    frames = downSample(videopath, 16)
    print(f'Downsample complete! {len(frames)} frames sampled.')
    messages = constructInputMessage(
        frames=frames,
        Prompt='What is going on in this video?',
        SysInstructions='You are a helpful assistant.',
        tempPath='./temp'
    )
    