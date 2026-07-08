#!/bin/bash
set -e

echo "=== Installing PyTorch CPU ==="
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Downloading model weights ==="
HF_BASE="https://huggingface.co/Bunny6397/AI-NST/resolve/main"

if [ ! -f "vgg_normalised.pth" ]; then
    wget -q -O vgg_normalised.pth "${HF_BASE}/vgg_normalised.pth"
    echo "Downloaded vgg_normalised.pth"
fi

mkdir -p experiment/final_exp
if [ ! -f "experiment/final_exp/decoder_final.pth" ]; then
    wget -q -O experiment/final_exp/decoder_final.pth "${HF_BASE}/decoder_final.pth"
    echo "Downloaded decoder_final.pth"
fi

echo "=== Build complete ==="
