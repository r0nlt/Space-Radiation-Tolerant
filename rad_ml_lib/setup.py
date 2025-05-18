from setuptools import setup, Extension, find_packages
import os
import platform
import sys
import pybind11

# Version information
VERSION = "0.1.0"

# Define C++ extension modules
ext_modules = []

cpp_extension = Extension(
    "rad_ml_lib.core._cpp_binding",
    sources=["rad_ml_lib/core/cpp_binding.cpp"],
    include_dirs=[
        pybind11.get_include(),
        "include",  # Include directory for header files
    ],
    language="c++",
    extra_compile_args=["-std=c++14", "-O3"],
)
ext_modules.append(cpp_extension)

# Package metadata
setup(
    name="rad_ml_lib",
    version=VERSION,
    author="Rishab Nuguru",
    author_email="your.email@example.com",
    description="Space Radiation-Tolerant Neural Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Space-Radiation-Tolerant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
        "dev": ["black", "flake8", "isort"],
    },
)
