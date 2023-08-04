from setuptools import find_packages, setup


setup(
    name="summaries",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "networkit",
        "networkx",
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
        "torch-geometric",
        "torch-scatter",
    ],
)
