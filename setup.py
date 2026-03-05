from setuptools import setup, find_packages

setup(
    name="crop_yield_prediction",
    version="0.0.1",
    author="Himadri",
    author_email="your_email@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)