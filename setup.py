from setuptools import setup, find_packages

setup(name="neuralnetwork", packages=find_packages())

# Notes:
# When developing a package - install into your virtual environment with:
# "pip install -e ." 
# Which lets you change source code om both test and application.
# Very useful for coverage testing
# https://docs.pytest.org/en/latest/goodpractices.html