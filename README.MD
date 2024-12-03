### HOW TO RUN?
docker pull vakidzaci/tritonserver-age-model:latest

docker run --rm --gpus all   -p 8000:8000 -p 8001:8001 -p 8002:8002  vakidzaci/tritonserver-age-model

python inference.py --img /path/to/img.jpg

python inference.py --folder /path/to/images


### triton server
Model is already converted into onnx format and saved inside image.


### download model weights

https://drive.google.com/file/d/1b70Jc5_XCN-9mE0eGM78Fbt0e8hzxjxX/view?usp=sharing

### model architecture
It is basically two resnet50 models that were trained in two ways.

First, train resnet50 model on three classes instead of four. Since, there is huge overlap between class 1 and class 2. Combine them into 1 class. All left is 0, 1/2, 3.

Second, train resnet50 model on two classes 1 and 2.

Use both weights on combined model. Freeze all the layers. Use them for Feature Extraction. Make classification on 4 classes.
