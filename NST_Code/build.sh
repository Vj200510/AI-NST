#!/bin/bash
set -e

# Install CPU-only torch first (smaller, faster — no CUDA needed on Render)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Download model weights from Hugging Face if not present
if [ ! -f "vgg_normalised.pth" ]; then
    echo "Downloading vgg_normalised.pth..."
    wget -q --show-progress -O vgg_normalised.pth \
        "https://huggingface.co/Bunny6397/AI-NST/resolve/main/vgg_normalised.pth"
    echo "vgg_normalised.pth downloaded."
fi

if [ ! -f "experiment/final_exp/decoder_final.pth" ]; then
    echo "Downloading decoder_final.pth..."
    mkdir -p experiment/final_exp
    wget -q --show-progress -O experiment/final_exp/decoder_final.pth \
        "https://huggingface.co/Bunny6397/AI-NST/resolve/main/decoder_final.pth"
    echo "decoder_final.pth downloaded."
fi

echo "Build complete."
