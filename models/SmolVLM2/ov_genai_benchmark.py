import time
import numpy as np
from PIL import Image
from datasets import load_from_disk
import openvino as ov
import openvino_genai as ov_genai

# [1. 配置参数]
model_path = "./SmolVLM2-500M-Video-Instruct-OV"
device = "GPU"  # 指定 Intel iGPU
num_samples = 10  # 建议先从小样本开始
target_output_len = 1024

# [2. 加载模型]
# openvino-genai 会自动处理所有的配置和 tokenizers
print(f"Loading model to {device} using OpenVINO GenAI...")
pipe = ov_genai.VLMPipeline(model_path, device)

# [3. 加载数据集]
print("Loading dataset...")
dataset = load_from_disk("../datasets/VQAv2_local")

# [4. 准备 Benchmark 数据]
latencies = []
output_token_counts = []

print(f"Starting Benchmark on {device} (Output: {target_output_len} tokens)...")

for i, example in enumerate(dataset.select(range(num_samples))):
    image_raw = example.get("image")
    if image_raw is None: continue
    
    # GenAI API 需要输入 ov.Tensor 格式的图片
    image_rgb = image_raw.convert("RGB")
    image_data = np.array(image_rgb)
    image_tensor = ov.Tensor(image_data)
    
    # 构造输入文字压力
    question = example.get("question", "")
    padding = " Please provide a very comprehensive and detailed analysis of this image. " * 30
    prompt = f"<ov_genai_image_0>{question} {padding}"

    # 配置生成参数
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = target_output_len
    config.min_new_tokens = target_output_len
    config.do_sample = False

    # 推理开始 (GenAI 接管了所有 Pre/Post-processing)
    start_time = time.perf_counter()
    
    # 直接调用 generate，支持传入 image 列表
    result = pipe.generate(prompt, images=[image_tensor], generation_config=config)
    
    end_time = time.perf_counter()
    
    # 统计
    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)
    
    # 注意：ov_genai.generate 直接返回字符串，需根据字符估算或通过 tokenizer 获取 token 数
    # 这里我们设定输出固定为 target_output_len 以符合你的 benchmark 逻辑
    output_token_counts.append(target_output_len)
    
    print(f"[{i+1}/{num_samples}] Latency: {latency_ms:.2f}ms | TPS: {target_output_len / (latency_ms/1000):.2f}")

# [5. 性能指标输出]
if latencies:
    avg_latency = sum(latencies) / len(latencies)
    total_gen_tokens = sum(output_token_counts)
    total_time = sum(latencies) / 1000

    print("\n" + "="*30)
    print(f"Device: {device} (OpenVINO GenAI API)")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Avg Throughput: {total_gen_tokens / total_time:.2f} tokens/sec")
    print(f"Target Output Length: {target_output_len} tokens")
    print("="*30)