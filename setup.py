from setuptools import setup, find_packages

setup(
    name="crop_yield_prediction",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.1",
        "scikit-learn>=1.3",
        "matplotlib>=3.8",
        "seaborn>=0.12",
        "joblib>=1.3",
        "mlflow>=2.5",
        "pyyaml>=6.0",
        "dvc>=2.64",
        "tqdm>=4.66",
        "optuna>=3.0"  
    ],
)