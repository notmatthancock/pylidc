from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

setup(
    name='pylidc',
    version='0.1.2',
    description='A library for working with the LIDC dataset.',
    long_description='',
    url='https://github.com/pylidc/pylidc',
    author='Matt Hancock',
    author_email='not.matt.hancock@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Database',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    keywords='pylidc LIDC lung sql research',
    packages=find_packages(exclude=['contrib', 'doc', 'tests*']),
    install_requires=[
        'sqlalchemy>=1.1.5', 'numpy>=1.12.0', 'scipy>=0.18.1',
        'matplotlib>=2.0.0', 'pydicom>=0.9.9', 'scikit-image>=0.12.3'
    ],
    package_data={
        'pylidc': ['pylidc.sqlite'],
    }
)
