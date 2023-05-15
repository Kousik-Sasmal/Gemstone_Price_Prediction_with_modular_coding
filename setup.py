from typing import List

from setuptools import setup
from setuptools import find_packages

HYPEN_E_DOT = '-e .'
def get_requirements():
    '''
    this function will return the list of requirements
    '''
    with open('requirements.txt','r') as f:
        requirements = f.readlines()

    requirements = [i.replace("\n","") for i in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="diamond-price-prediction",
    version='0.0.1',
    #description="Diamond Price Prediction",
    author='Kousik Sasmal',
    author_email='kousik712417@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements()
)
