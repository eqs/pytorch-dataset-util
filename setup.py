import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'),
                      text_type(r'``\1``'), fd.read())


def read_requirements(filename):
    with open(filename, 'r') as f:
        requirements = list(map(lambda x: x[:-1], f.readlines()))
    return requirements


setup(
    name="pytorch-dataset-util",
    version="0.0.1",
    url="https://github.com/eqs/pytorch-dataset-util.git",
    license='MIT',

    author="eqs",
    author_email="murashige.satoshi.mi1 [at] is.naist.jp",

    description="Dataset classes",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    install_requires=read_requirements("requirements.txt"),

    extras_require={
        'docs': ['sphinx >= 2.2.0']
    }

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
