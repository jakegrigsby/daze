from setuptools import setup, find_packages

setup(name='deepzip',
      version='0.0.1',
      setup_requires=["pytest-runner",],
      tests_require=["pytest",],
      description='Deep Learning for Compression',
      url='http://github.com/jakegrigsby/deepzip',
      author='Jake Grigsby and Jack Morris',
      author_email='jcg6dn@virginia.edu',
      license='MIT',
      packages=find_packages(),)
