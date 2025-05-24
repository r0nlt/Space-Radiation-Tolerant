#!/usr/bin/env python3


import os
import re
import sys
import platform
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_BINDINGS=ON",
        ]

        # Pile driver options for the compiler
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -std=c++17'

        build_dir = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_dir, exist_ok=True)

        # Build the CMake project
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_dir, env=env
        )
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)


# Get version from rad_ml C++ header
def get_version():
    version_file = "../include/rad_ml/api/rad_ml.hpp"
    version_pattern = r"static constexpr int (major|minor|patch) = (\d+);"

    version = {}
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            content = f.read()
            for match in re.finditer(version_pattern, content):
                version[match.group(1)] = match.group(2)

        if "major" in version and "minor" in version and "patch" in version:
            return f"{version['major']}.{version['minor']}.{version['patch']}"

    return "2.0.0"  # Default version


# Get long description from README.md or use a default
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text()
    long_description_content_type = "text/markdown"
else:
    long_description = """
    # Radiation-Tolerant Machine Learning Framework (Python)

    Python bindings for the C++ rad_ml framework, enabling radiation-tolerant
    machine learning in Python applications. The framework is designed to protect
    machine learning models running in radiation environments, such as space.
    """
    long_description_content_type = "text/markdown"


setup(
    name="rad_ml",
    version=get_version(),
    author="Rishab Nuguru",
    author_email="rishabnuguru@example.com",
    description="Radiation-Tolerant Machine Learning Framework",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url="https://github.com/r0nlt/Space-Radiation-Tolerant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: AGPL v3 license",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Space",
    ],
    python_requires=">=3.7",
    ext_modules=[CMakeExtension("rad_ml._core")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": ["pytest", "flake8", "mypy"],
    },
)
