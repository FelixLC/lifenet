#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='lifenet',
      version='0.0.1',
      description='Lifenet package: a deep learning library thanks to Joel',
      author='Lifen',
      author_email='user@email.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
    )