from setuptools import setup, find_packages

setup(
    name="transformer_architecture_implementation",
    version="0.1.0",
    packages=find_packages(
        include=["transformer_architecture", "transformer_architecture.*"]
    ),
    description="Python programm for creating an implementation\
        of the Transformer architecture",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
)
