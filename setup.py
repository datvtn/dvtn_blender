from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("dvtn_blender.dvtn_blender", ["dvtn_blender/laplacian_blending.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="Datvtn Blender",
    version="0.1",
    author="Dat Viet Thanh Nguyen",
    description="A Collection of Blender Utilities",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=[
        "Cython",
        "numpy",
        "opencv-python"
    ],
)
