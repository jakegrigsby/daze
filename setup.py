from setuptools import setup, find_packages

setup(
    name="daze",
    version="0.0.1",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    description="AutoEncoder library for reproducible research.",
    url="http://github.com/jakegrigsby/daze",
    author="Jake Grigsby and Jack Morris",
    author_email="jcg6dn@virginia.edu",
    license="MIT",
    packages=find_packages(),
)
