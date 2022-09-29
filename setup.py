import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='scevonet',
     version='0.0.8',
     author="Aleksandr Kotov",
     author_email="alexander.o.kotov@gmail.com",
     description="Tool for generation [cell state - gene program] network",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/Qotov/scEvoNet",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    install_requires=['lightgbm>=3.0.0', 'pandas>=0.24', 'networkx>=2.5']
 )
