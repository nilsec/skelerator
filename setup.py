from setuptools import setup

setup(
    name='skelerator',
    version='0.1',
    description='Neuronal data generator',
    url='https://github.com/nilsec/mtrack',
    author='Nils Eckstein',
    author_email='ecksteinn@hhmi.org',
    license='MIT',
    packages=[
        'skelerator',
            ],
    install_requires = [
        'numpy',
        'scipy',
        'h5py',
        'mahotas',
            ],
)   
