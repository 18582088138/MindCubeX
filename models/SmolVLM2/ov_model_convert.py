import torch
from transformers import AutoProcessor
from optimum.intel import OVModelForVisualCausalLM # 必须使用 Visual 类
from PIL import Image
import requests

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
local_path = "SmolVLM2-500M-Video-Instruct-OV"

# 1. 下载并转换模型 (一步到位)
# export=True 会调用 Optimum 的导出逻辑，自动处理 Vision + Language 部分
print("[OV Logging] Converting and Exporting to OpenVINO IR...")
model = OVModelForVisualCausalLM.from_pretrained(
    model_id, 
    export=True, 
    device="CPU",        # 转换过程建议先用 CPU
    load_in_8bit=False,  # 如果需要量化，可以改为 True 或使用 NNCF
    trust_remote_code=True
)

# 2. 保存转换后的 IR 模型和 Processor
model.save_pretrained(local_path)
processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(local_path)
print(f"[OV Logging] Model and Processor saved at: {local_path}")

# --- 性能测试 (Benchmark) 部分 ---

# 3. 重新加载模型到 GPU (Intel GPU)
print(f"[OV Logging] Loading model to GPU...")
ov_model = OVModelForVisualCausalLM.from_pretrained(
    local_path, 
    device="GPU",
    ov_config={"PERFORMANCE_HINT": "LATENCY"}
)

# 4. 准备多模态输入 (以一张图片为例)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 使用聊天模板构建 Prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# 5. 推理性能测试
# VLM 的 inputs 需要由 processor 生成，包含 pixel_values
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cpu")

# print("[OV Logging] Starting Inference...")
# import time
# start = time.time()

# # 模拟 1k token 输出的压力测试
# output_ids = ov_model.generate(
#     **inputs, 
#     max_new_tokens=100, # 实际测试时改为 1024
#     min_new_tokens=100, 
#     do_sample=False
# )

# end = time.time()

# generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)
# print(f"Output: {generated_text[0]}")
# print(f"Latency for 100 tokens: {(end - start):.2f}s")


# [4Bit Compression]
from optimum.intel import OVWeightQuantizationConfig

local_path = "SmolVLM2-500M-Video-Instruct-INT4-OV"

q_config = OVWeightQuantizationConfig(
        bits=4,
        group_size=64,
        sym=False,
        ratio=0.8,)

print("Quantizing model to INT4...")
model = OVModelForVisualCausalLM.from_pretrained(
    model_id,
    export=True,
    quantization_config=q_config,
    trust_remote_code=True
)

model.save_pretrained(local_path)
processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(local_path)
print(f"[OV Logging] Model Compress success and saved at: {local_path}")