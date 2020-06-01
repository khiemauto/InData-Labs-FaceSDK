import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="face_recognition_sdk",
    version="0.0.1",
    author="InData Labs",
    author_email="info@indatalabs.com",
    description="A python package for face detection and recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.indatalabs.com/cv_rg/face-recognition-sdk",
    packages=["face_recognition_sdk"],
    python_requires=">=3.7",
)
