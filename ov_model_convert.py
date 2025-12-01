import os
import sys
import logging

from pathlib import Path
from cmd_helper import optimum_cli

import numpy as np
import torch
import openvino

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

model_id = "Qwen/Qwen3-VL-8B-Instruct"
output_dir = Path(model_id.split("/")[-1])

to_compress = "FP16"

additional_args = {}

if to_compress == "INT4":
    output_dir = output_dir / "INT4_asym_g128_r0.8"
    additional_args.update({"task": "image-text-to-text", "weight-format": "int4", "group-size": "128", "ratio": "0.8"})
elif to_compress == "INT8":
    output_dir = output_dir / "INT8_asym"
    additional_args.update({"task": "image-text-to-text", "weight-format": "int8"})
else : 
    output_dir = output_dir / "FP16"
    additional_args.update({"task": "image-text-to-text", "weight-format": "fp16"})

if not output_dir.exists():
    logger.debug(f"[OV] {model_id} model convert start ...")
    optimum_cli(model_id, output_dir, additional_args=additional_args)
    logger.info(f"[OV] {model_id} model convert success, saving at {output_dir}")

logger.info(f"[OV] Model convert task done!")