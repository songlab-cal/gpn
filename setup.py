from setuptools import setup


install_requires = [
    "transformers[torch]",
    "datasets",
    "pyarrow",
    "pandas",
    "numpy",
    "torchinfo",
    "biopython",
    "wandb",
    "einops",
    "pandarallel",
    "bioframe",
    "zstandard",
    "zarr",
    "pyBigWig",
    "joblib",
]


setup(
    name='gpn',
    version='0.6',
    description='gpn',
    url='http://github.com/songlab-cal/gpn',
    author='Gonzalo Benegas',
    author_email='gbenegas@berkeley.edu',
    license='MIT',
    packages=['gpn', 'gpn.ss', 'gpn.msa'],
    zip_safe=False,
    install_requires=install_requires,
)
