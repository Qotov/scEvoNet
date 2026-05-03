import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scevonet",
    version="2.0.0",
    author="Aleksandr Kotov",
    author_email="alexander.o.kotov@gmail.com",
    description="Cell state and gene program networks from scRNA-seq (LightGBM / cross-dataset)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Qotov/scEvoNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "lightgbm>=3.0.0",
        "pandas>=1.3",
        "numpy>=1.20",
        "networkx>=2.5",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "seaborn>=0.11",
    ],
)
