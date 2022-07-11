from setuptools import setup

setup(
    name="pydebiaseddta",
    version="0.0.3",
    description="Python library to improve generalizability of the drug-target prediction models via DebiasedDTA",
    url="https://github.com/rizaozcelik/pydebiaseddta",
    author="Rıza Özçelik",
    author_email="riza.ozcelik@boun.edu.tr",
    license="MIT",
    python_requires=">=3.9.7",
    install_requires=[
        "numpy==1.21.2",
        "scikit_learn==1.1.1",
        "tensorflow==2.7.0",
        "tokenizers==0.10.3",
        "transformers==4.14.1",
    ],
    include_package_data=True,
    zip_safe=False,
)

