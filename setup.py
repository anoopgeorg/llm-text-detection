from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def getRequirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [
            line.replace("\n", "") for line in requirements if line != HYPHEN_E_DOT
        ]
    return requirements


setup(
    name="llm-text-detection",
    version="0.0.1",
    author="Anoop George",
    author_email="anoopgeorge1126@gmail.com",
    packages=find_packages(),
    install_requires=getRequirements("requirements.txt"),
)
