"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

about = {}
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'pretorched', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    url=about['__url__'],                  # Optional
    name=about['__title__'],               # Required
    author=about['__author__'],            # Optional
    version=about['__version__'],          # Required
    keywords=about['__keywords__'],        # Optional
    author_email=about['__email__'],       # Optional
    description=about['__description__'],  # Required
    long_description=long_description,     # Optional
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['torch', 'torchvision', 'munch'],    # Optional
    packages=find_packages(exclude=['data', 'examples']),  # Required
)
