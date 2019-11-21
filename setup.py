from setuptools import setup, find_packages

setup(
    name='GILC',
    version='0.1.0dev',
    author='Alexandra Gessner',
    author_email='agessner@tue.mpg.de',
    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    license='LICENSE.txt',
    description='Integrals of Gaussians under linear domain constraints',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.15.4",
    ],
)
