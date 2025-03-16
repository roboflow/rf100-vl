from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="rf100vl",
    version="0.1.0",
    description="RF100VL Dataset Interface",
    author="Peter Robicheaux",
    author_email="peter@roboflow.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)