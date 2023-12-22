from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
__version__ = "0.0.1"
REPO_NAME = "llm-text-detection"
AUTHOR_USER_NAME = "anoopgeorg"
SRC_REPO = "llm-text-detection"
AUTHOR_EMAIL = "anoopgeorge1126@gmail.com"


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
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=getRequirements("requirements.txt"),
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
)
