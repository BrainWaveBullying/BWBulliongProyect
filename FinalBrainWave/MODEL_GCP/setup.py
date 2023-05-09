from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='brainwave-BullyingPredict',
    version='0.1.0',
    author='Brain Wave',
    description='Este proyecto tiene como objetivo detectar casos de bullying en base a una encuesta',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BrainWaveBullying/BullyingProject',
    packages=find_packages(),
    install_requires=[
        'apache-beam[gcp]==2.24.0'
        'numpy>=1.20.0',
        'pandas>=1.3.3',
         ],
    python_requires='>=3.7',
)