from PIL import Image
from transformers import AutoProcessor
from transformers import TextStreamer

from pathlib import Path
import requests

from optimum.intel.openvino import OVModelForVisualCausalLM

device = "GPU"
model_dir = "Qwen3-VL-8B-Instruct/FP16"

model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

# if processor.chat_template is None:
#     tok = AutoTokenizer.from_pretrained(model_dir)
#     processor.chat_template = tok.chat_template

example_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
example_image_path = Path("demo.jpeg")

if not example_image_path.exists():
    Image.open(requests.get(example_image_url, stream=True).raw).save(example_image_path)

image = Image.open(example_image_path)
question = "Describe this image."

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": f"{example_image_url}",
            },
            {"type": "text", "text": question},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")

# display(image)
print("Question:")
print(question)
print("Answer:")

generated_ids = model.generate(**inputs, max_new_tokens=100, streamer=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True))