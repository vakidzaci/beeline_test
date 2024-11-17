### HOW TO RUN?
docker pull vakidzaci/tritonserver-age-model:latest

docker run --rm --gpus all   -p 8000:8000 -p 8001:8001 -p 8002:8002  vakidzaci/tritonserver-age-model

python inference.py --img /path/to/img.jpg

python inference.py --folder /path/to/images


### triton server
Model is already converted into onnx format and saved inside image.