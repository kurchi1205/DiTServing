from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name='glideServe',
    version='0.1',
    packages=find_packages(),  # Automatically finds all submodules
    install_requires=read_requirements(),       # You can add your dependencies here or use requirements.txt
    include_package_data=True,
)
