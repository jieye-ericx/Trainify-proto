import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Trainify-proto",
    version="0.1.1",
    author="ericx",
    author_email="ericxlee@formail.com",
    description="Trainify-proto",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jieye-ericx/Trainify-proto",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch==1.10.0',
        'pandas==1.3.4',
        'psutil==5.8.0',
        'Rtree==0.9.7',
        'lark==0.11.3',
        'gym==0.23.0',
        'matplotlib==3.4.3',
        'numpy==1.20.0',
        'scipy==1.8.1',
        'tensorboardX==2.5.1'
    ]
)
