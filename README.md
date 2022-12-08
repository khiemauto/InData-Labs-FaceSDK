## Installation

Clone and install dependencies for development

```
git clone https://github.com/khiemauto/InData-Labs-FaceSDK.git
cd InData-Labs-FaceSDK
conda env create -f environment.yml
conda activate face_sdk

Copy image with name to demo/employees/images

cd demo
python recognition_on_image.py -p <Photo Test>
<!-- pre-commit install -->
<!-- git lfs install -->
<!-- python setup.py build develop -->
```

install only face_recognition_sdk with dependencies

```
pip install .
```