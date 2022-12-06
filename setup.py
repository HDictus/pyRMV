import imp
from setuptools import setup, find_packages

VERSION = imp.load_source("", "analysis_neuro/version.py").VERSION

with open('README.rst', encoding='utf-8') as f:
    README = f.read()

setup(
    name="analysis-neuro",
    author="bbp",
    version=VERSION,
    long_description=README,
    long_description_content_type="text/x-rst",
    description="collection of analyses and validations for neuroscience",
    license="MIT",
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'scipy'
    ],
    extras_require={
        "docs": ["sphinx", "sphinx-bluebrain-theme"]},
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/browse/MMBVISCX-19",
        "Source": "https://bbpgitlab.epfl.ch/circuits/personal/analysis-neuro",
    },
    url="http://bluebrain.epfl.ch",
    packages=find_packages(),
    classifiers=[
        'Development Status :: Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
