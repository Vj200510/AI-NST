#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Download model weights if not present
# Replace the URLs below with your actual Google Drive / Hugging Face direct download links

if [ ! -f "vgg_normalised.pth" ]; then
    echo "Downloading vgg_normalised.pth..."
    # Example: wget -O vgg_normalised.pth "YOUR_DIRECT_DOWNLOAD_URL"
fi

if [ ! -f "experiment/final_exp/decoder_final.pth" ]; then
    echo "Downloading decoder_final.pth..."
    mkdir -p experiment/final_exp
    # Example: wget -O experiment/final_exp/decoder_final.pth "YOUR_DIRECT_DOWNLOAD_URL"
fi
