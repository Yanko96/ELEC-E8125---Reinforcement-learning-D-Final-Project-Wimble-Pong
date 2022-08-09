import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="a2c-wimblepong",
    version="1.0",
    author="Yangzhe Kong",
    author_email="yangzhe.kong@hotmail.com",
    description="A2C Solution to Wimblepong Game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Yanko96/ELEC-E8125---Reinforcement-learning-D-Final-Project-Wimble-Pong",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
