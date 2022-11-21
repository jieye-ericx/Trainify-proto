import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Trainify-proto",
    version="0.1.8",
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
        'torch',
        'numpy',
        'scipy',
        'gym',
        'colorlog',
        'tensorboardx',
        'pandas',
        'rtree',
        'matplotlib',
        'lark',
        'setuptools'
    ]
)
