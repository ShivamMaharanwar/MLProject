from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        requirements = [req for req in requirements if req != HYPEN_E_DOT]
    return requirements

setup(
    name="MLProject",
    version="0.1.0",
    author="shivam",
    author_email="shivammaharanwar123@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)