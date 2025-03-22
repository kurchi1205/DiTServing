from setuptools import setup, find_packages

setup(
    name='custom_serve',
    version='0.1',
    packages=find_packages(),  # Automatically finds all submodules
    install_requires=[],       # You can add your dependencies here or use requirements.txt
    include_package_data=True,
    description='My custom package with submodules',
)
