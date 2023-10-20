from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:

    '''
    this function will return list of requirements


    :param file_path:
    :return:list of requirements
    '''
    requirements=[]
    hypen_dot='-e .'
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',"") for req in requirements]

        if hypen_dot in requirements:
            requirements.remove(hypen_dot)







    return requirements
setup(
    name='True Value Car price estimator',
    version='0.0.1',
    author='adhi',
    author_email='adhithyagnair97@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
