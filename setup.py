from setuptools import setup
from setuptools import find_packages
setup(name='plexusnet',
version='0.1',
description='PlexusNet for medical image analysis',
url='https://github.com/oeminaga/PlexusNet.git',
author='Okyaz Eminaga',
author_email='okyaz.eminaga@gmail.com',
install_requires=['keras>=2.3.1', 'numpy>=1.9.1'],
license='GPL. Please cite when you use, replicate or modfiy the package.',
packages=find_packages(),
zip_safe=False)
