from setuptools import setup, find_packages

setup(
    name='equivariant-ip',
    author='Ferdia Sherry',
    author_email='fs436@cam.ac.uk',
    packages=find_packages(),
    license='GPLv3',
    version='0.0.1',
    install_requires=['torch', 'e2cnn', 'numpy', 'scikit-image'],
    extras_require={'ct': 'astra'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
    ],
)
