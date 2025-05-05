from setuptools import setup, find_packages

setup(
    name="tablesense",                # The name of your package
    version="0.1.0",                  # Your package version
    packages=find_packages(include=['tablesense', 'tablesense.*']),  # Tells Python which folders to include
    install_requires=[                # List of packages your code needs
        'torch>=1.9.0',
        'pandas>=1.3.0',
        'openpyxl>=3.0.7',
        'numpy>=1.19.0',
        'tqdm>=4.62.0',
    ],
    python_requires='>=3.7',          # Minimum Python version required
)