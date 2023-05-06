from setuptools import find_packages,setup    # find all the packages avaliable in the in the directory we have created or application

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str):
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    
    with open(file_path,encoding='utf-8') as file_obj:
        requirements=file_obj.read()
        requirements=requirements.replace("\n"," ").split(" ")
        '''
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        '''
        requirements=requirements[:-2]
    return requirements  

#all basic information about the project or metadata information
setup(
    name='mlproject1',
    version='0.0.1',
    author='N.Jashwanth Reddy',
    author_email='tangwulinggd@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')# or ['pandas','numpy','sklearn']. all the required packages
)