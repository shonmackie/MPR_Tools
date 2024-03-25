import setuptools

with open("README.txt", 'r') as f:
    long_description=f.read()

setuptools.setup(
    name='MPR_Tools',
    version='1.0',
    description='Toolkit for analyzing MPR Neutron Spectrometers',
    license='MIT',
    lonbg_description=long_description,
    author='Shon Mackie',
    author_email='smackie@mit.edu',
    packages=setuptools.find_packages(),
)