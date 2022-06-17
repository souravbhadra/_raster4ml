import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="raster4ml",
    version="0.0.1",
    author="Sourav Bhadra",
    description="A geospatial raster processing library for machine learning",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',    
    py_modules=["raster4ml"],    
    #package_dir={'':'raster4ml'},
    install_requires=read('requirements.txt').splitlines() 
)