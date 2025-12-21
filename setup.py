"""Setup script for steering-awareness package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="steering-awareness",
    version="0.1.0",
    author="",
    author_email="",
    description="Training LLMs to detect activation steering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "eval": ["google-genai>=0.3.0"],
        "dev": ["pytest", "black", "isort", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "train-steering=experiments.run_training:main",
            "eval-steering=experiments.run_evaluation:main",
        ],
    },
)
