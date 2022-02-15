import imp
from setuptools import setup, find_packages

VERSION = imp.load_source("", "analysis_library/version.py").__version__

with open('README.rst', encoding='utf-8') as f:
    README = f.read()

setup(
    name="analysis-library",
    author="bbp",
    version=VERSION,
    long_description=README,
    long_description_content_type="text/x-rst",
    description="collection of analyses and validations for neuroscience",
    license="MIT",
    python_requires='>=3.6',
    packages=find_packages(),
    classifiers=[
        'Development Status :: Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
