import time
import torch
import numpy as np
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor
from optimum.intel import OVModelForVisualCausalLM

# [1. 配置参数]
# model_path = "./SmolVLM2-500M-Video-Instruct-OV"
model_path = "./SmolVLM2-500M-Video-Instruct-INT4-OV"
device = "GPU"  # 使用 Intel GPU
batch_size = 8
num_samples = 100  # 测试样本数，建议先设为 10-20 验证速度
target_input_len = 1024
target_output_len = 1024

# OpenVINO GPU 性能配置
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_DIR": "./ov_cache",
    "INFERENCE_PRECISION_HINT": "f16",

    # "GPU_THROUGHPUT_STREAMS": "1",
}

# [2. 加载模型与 Processor]
print(f"Loading model to {device}...")
model = OVModelForVisualCausalLM.from_pretrained(
    model_path,
    device=device,
    ov_config=ov_config,
    trust_remote_code=True,
    compile=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.tokenizer.padding_side = 'left'
# [3. 加载数据集]
print("Loading VQAv2 dataset...")
dataset = load_from_disk("../datasets/VQAv2_local")


# [4. 性能测试循环]
ftl_list = []      # First Token Latency (ms)
ntl_list = []      # Next Token Latency (ms/token)
total_latencies = []

print(f"Starting Benchmark (Input: ~1k, Output: 1k)...")

for i in range(0, num_samples, batch_size): # 测试前 100 个样本
    batch_examples = dataset.select(range(i, min(i + batch_size, len(dataset))))
    current_bs = len(batch_examples)
    
    # 准备批数据
    images = [[ex['image'].convert("RGB")] for ex in batch_examples]
    padding_text = " Please analyze the image content in extreme detail. " * 50
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex['question'] + padding_text}]}],
            add_generation_prompt=True
        ) for ex in batch_examples
    ]
    
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    input_len = inputs["input_ids"].shape[1]

    # --- 核心统计逻辑 ---
    
    # 我们使用自定义循环或回调太复杂，这里用简单的生成 1 个 token 测 FTL，
    # 再生成剩余 token 测 NTL 的方式虽有微量误差但最直观。
    
    # 1. 测量 Prefill + First Token
    t0 = time.perf_counter()
    out_first = model.generate(**inputs, max_new_tokens=1, use_cache=True)
    t1 = time.perf_counter()
    ftl = (t1 - t0) * 1000
    
    # 2. 测量后续 Decoding
    # 为了连贯性，我们将第一个 token 结果作为输入继续
    t2 = time.perf_counter()
    out_full = model.generate(
        **inputs, 
        max_new_tokens=target_output_len,
        min_new_tokens=target_output_len,
        use_cache=True,
        do_sample=False
    )
    t3 = time.perf_counter()
    
    # 计算 NTL: 总解码时间 / 生成的 token 数
    decoding_time_ms = (t3 - t2) * 1000
    ntl = decoding_time_ms / (target_output_len - 1)
    

    ftl_list.append(ftl)
    ntl_list.append(ntl)
    total_latencies.append(ftl + decoding_time_ms)

    print(f"[{i+1:02d}] Batch:{current_bs} | E2E Latency: {(ftl + decoding_time_ms)/1000:.2f} s | FTL: {ftl:.2f}ms | NTL: {ntl:.2f}ms/tk | Decoding TPS: {1000/ntl*current_bs:.2f}")
    print(f"     Per Sample | E2E Latency : {((ftl + decoding_time_ms)/current_bs)/1000:.2f} s | FTL: {(ftl/current_bs):.2f} ms | NTL: {(ntl/current_bs):.2f} ms/tk")

# [5. 最终统计]
avg_ftl = np.mean(ftl_list)
avg_ntl = np.mean(ntl_list)
avg_tps = 1000 / avg_ntl

print("\n" + "="*40)
print(f"Device: {device} | Model: SmolVLM2-500M-INT4")
print(f"Average First Token Latency (TTFT): {avg_ftl:.2f} ms")
print(f"Average Next Token Latency (TPOT):  {avg_ntl:.2f} ms/token")
print(f"Average Decoding Throughput:       {avg_tps:.2f} tokens/sec")
print(f"Total Response Time (1k out):      {np.mean(total_latencies)/1000:.2f} s")
print("="*40)