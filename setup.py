from setuptools import setup, find_packages


def main():
    setup(
        name='pylidc',
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
            'matplotlib>=2.0.0', 'pydicom>=1.0.0', 'scikit-image>=0.13'
        ],
        package_data={
            'pylidc': ['pylidc.sqlite'],
        }
    )


if __name__ == '__main__':
    main()
