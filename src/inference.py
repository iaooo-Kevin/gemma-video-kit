from transformers import AutoProcessor, AutoModelForImageTextToText, Gemma3ForConditionalGeneration, Gemma3nForConditionalGeneration
from PIL import Image
import PIL
import torch
from typing import List, Dict, Optional, Union, Tuple
import requests
from video_utils import downSample, constructInputMessage

def runInference(modelID: str, messages: List[Dict[str, Union[str, PIL.Image.Image]]], temperature: float = 0.7, max_new_tokens: int = 128) -> str:
    """
    Performs inference using the Gemma class of models. Currently available models are:
    `google/gemma-3-4b-it`, `google/gemma-3-1b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
    `google/gemma-3n-E4B-it`, `google/gemma-3n-E2B-it`

    Arguments:
        modelID (str): The Huggingface model ID to use for inference.
        messages (List[Dict[str, Union[str, PIL.Image.Image]]]): The input messages to be processed.
        temperature (float): The temperature to use for sampling. Defaults to 0.7.
        max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 128.
    Returns:
        str: The model output (generated text)
    """
    if modelID not in ["google/gemma-3-4b-it", "google/gemma-3-1b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it", "google/gemma-3n-E4B-it", "google/gemma-3n-E2B-it"]:
        raise ValueError("Invalid model ID. Please choose from the available models.")
    if "3n" in modelID:
        try:
            model = Gemma3nForConditionalGeneration.from_pretrained(modelID, device_map='auto', torch_dtype=torch.bfloat16).eval()
            processor = AutoProcessor.from_pretrained(modelID, use_fast=True)

        except Exception as e:
            if "invalid credentials" in str(e).lower():
                raise ValueError('You are required to first login via HuggingFace, please do so using `huggingface-cli login` and paste your token. Either that, or you might be trying to access a gated model, in that case, please get access in that case.') 
            else:
                raise ValueError(f"An error occurred while loading the model: {e}")
    else:
        try:
            model = Gemma3ForConditionalGeneration.from_pretrained(modelID, device_map='auto', torch_dtype=torch.bfloat16).eval()
            processor = AutoProcessor.from_pretrained(modelID, use_fast=True)
        except Exception as e:
            if "invalid credentials" in str(e).lower():
                raise ValueError('You are required to first login via HuggingFace, please do so using `huggingface-cli login` and paste your token. Either that, or you might be trying to access a gated model, in that case, please get access in that case.')
            else:
                raise ValueError(f"An error occurred while loading the model: {e}")
    
    Inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        tokenize=True,
        return_tensors='pt'
    )
    Inputs = Inputs.to(model.device, dtype=torch.bfloat16)
    
    inputLen = Inputs['input_ids'].shape[-1]
    with torch.inference_mode():
        generate = model.generate(**Inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
        output = generate[0][inputLen:]
    
    #Decode the output.
    decoded = processor.decode(output, skip_special_tokens=True)
    return decoded


if __name__ == "__main__":
    #Example usage for videos.
    videopath = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
    frames = downSample(videopath, 16)
    print(f'Downsample complete! {len(frames)} frames sampled.')
    messages = constructInputMessage(
        frames=frames,
        Prompt='What is going on in this video?',
        SysInstructions='You are a helpful assistant.',
        tempPath='./temp'
    )
    output = runInference(
        modelID = 'google/gemma-3-4b-it',
        messages=messages,
        max_new_tokens=128,
    )
    print(f'Model Output: {output}')