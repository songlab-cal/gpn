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
]


setup(
    name='gpn',
    version='0.1',
    description='gpn',
    url='http://github.com/songlab-cal/gpn',
    author='Gonzalo Benegas',
    author_email='gbenegas@berkeley.edu',
    license='MIT',
    packages=['gpn', 'gpn.mlm', 'gpn.chromatin', 'gpn.data', 'gpn.msa'],
    zip_safe=False,
    install_requires=install_requires,
)
