from setuptools import setup,find_packages
from typing import List

def get_requirements(filepath:str) -> List[str]:
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")
        
        return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Abhishek Kadam",
    author_email="abhikads444@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(r"C:\Users\Abhishek\Desktop\ML\mlproject\requirements.txt")
)