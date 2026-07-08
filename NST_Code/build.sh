#!/bin/bash
set -e

# Install CPU-only torch + torchvision
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

HF_BASE="https://huggingface.co/Bunny6397/AI-NST/resolve/main"

# Download vgg_normalised.pth
if [ ! -f "vgg_normalised.pth" ]; then
    echo "Downloading vgg_normalised.pth (76MB)..."
    wget --tries=3 --timeout=300 -q --show-progress \
        -O vgg_normalised.pth "${HF_BASE}/vgg_normalised.pth"
    echo "Done: vgg_normalised.pth"
fi

# Download decoder_final.pth
mkdir -p experiment/final_exp
if [ ! -f "experiment/final_exp/decoder_final.pth" ]; then
    echo "Downloading decoder_final.pth (13MB)..."
    wget --tries=3 --timeout=300 -q --show-progress \
        -O experiment/final_exp/decoder_final.pth "${HF_BASE}/decoder_final.pth"
    echo "Done: decoder_final.pth"
fi

echo "=== Build complete ==="
ls -lh vgg_normalised.pth experiment/final_exp/decoder_final.pth
