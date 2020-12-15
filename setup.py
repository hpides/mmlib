import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements-setup.txt') as f:
    install_requires = [l.strip() for l in f]

tests_require = [
    'torchvision'
]

setuptools.setup(
    name="mmlib",
    version="0.0.1",
    author="Nils Strassenburg",
    description="library for model management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="PUT URL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
    },
    python_requires='>=3.6',
)
