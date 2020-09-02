import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="object-detection-fastai", 
    version="0.0.9",
    author="Christian Marzahl",
    author_email="christian.marzahl@gamil.com",
    description="Uses RetinaNet with FastAi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChristianMarzahl/ObjectDetection",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)