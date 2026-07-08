FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller image)
RUN pip install --no-cache-dir torch==2.5.0 torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Download model weights from Hugging Face
RUN wget -q -O vgg_normalised.pth \
    "https://huggingface.co/Bunny6397/AI-NST/resolve/main/vgg_normalised.pth" && \
    mkdir -p experiment/final_exp && \
    wget -q -O experiment/final_exp/decoder_final.pth \
    "https://huggingface.co/Bunny6397/AI-NST/resolve/main/decoder_final.pth"

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["python", "app.py"]
