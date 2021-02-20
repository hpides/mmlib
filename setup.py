import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]

tests_require = []

setuptools.setup(
    name="mmlib",
    version="0.0.1",
    author="Nils Strassenburg",
    author_email="Nils.Strassenburg@student.hpi.uni-potsdam.de",
    description="library for model management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slin96/mmlib",
    packages=['mmlib', 'util', 'schema'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
    },
    python_requires='>=3.8',
)
