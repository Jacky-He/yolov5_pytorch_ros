from setuptools import setup 
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    version="0.0.1",
    packages=['utils', 'models'],
    package_dir={'': 'scripts'}
)

setup(**setup_args)