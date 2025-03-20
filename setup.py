from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rf100vl",
    version="1.0.0",
    description="RF100-VL Dataset Interface",
    author="Roboflow, Inc.",
    author_email="peter@roboflow.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "rf100vl": ["assets/*.json"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="machine-learning, deep-learning, computer-vision, ML, DL, CV, AI"
)