#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()


setup(
    name="adaptor",
    version='0.2.5',
    description="Adaptor: Objective-centric Adaptation Framework for Language Models.",
    long_description_content_type="text/markdown",
    long_description=readme,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8"
    ],
    author="Michal Stefanik & Adaptor Authors & Contributors",
    author_email="stefanik.m@mail.muni.cz",
    url="https://github.com/gaussalgo/adaptor",
    python_requires=">=3.7",
    license="MIT",
    packages=find_packages(include=["adaptor", "adaptor.*"]),
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        "torch>=1.7",
        "transformers<=4.30.2",  # TODO upper-closed on 4.30.2: Problem with returning empty batches
        "sentencepiece",
        "accelerate>=0.20.1",
        "peft>=0.10.0,<0.13.0",
        "prefetch-generator>=1.0.3",
        "numpy<1.24"  # constrained by integration of a prism metric, can be removed, once prism is deprecated
    ],
    test_require=[
        "pytest"
    ],
    extras_require={
        "generative": [
            "sacrebleu",
            "rouge-score",
            "bert-score",
            "fairseq",
            "protobuf<=3.20.1",
            # "omegaconf>=2.2"  # previous versions are incompatible with pip<25 for unsupported deps syntax ('>=5.1.*')
        ],
        "examples": [
            "comet-ml",
        ],
    },
)
