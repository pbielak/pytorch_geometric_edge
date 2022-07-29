from setuptools import find_packages, setup

with open("README.md", "r") as fin:
    long_description = fin.read()


setup(
    name='pytorch_geometric_edge',
    version='0.0.1',
    url='',
    license='MIT',
    author='Piotr Bielak',
    author_email='piotr.bielak@pwr.edu.pl',
    description='Edge representation learning library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.10.0',
        'torch-geometric>=2.0.4',
    ],
    extras_require={
        'test': ['pytest'],
    },
    packages=find_packages(),
)
