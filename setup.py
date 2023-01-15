from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="edf_env",
    version="0.0.1",
    author="Hyunwoo Ryu",
    author_email="tomato1mule@gmail.com",
    description="Pybullet environment for EDF.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomato1mule/edf_env",
    project_urls={
        "Bug Tracker": "https://github.com/tomato1mule/edf_env/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 22.04",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        'numpy==1.24.1',
        'scipy==1.10.0',
        'pybullet==3.2.5',
        'pandas==1.5.2',
        'pyyaml',        # 6.0
        'tqdm',          # 4.64.1
        'jupyter',       # 1.0.0
        'plotly',        # 5.12.0
        'mypy',
        'nb_mypy',
        'types-PyYAML',
    ]
)