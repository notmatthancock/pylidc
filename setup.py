from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

setup(
    name='pylidc',
    version='0.1.0',
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
        'sqlalchemy', 'numpy', 'scipy',
        'matplotlib', 'pydicom', 'scikit-image'
    ],
    package_data={
        'pylidc': ['pylidc.sqlite'],
    }
)
