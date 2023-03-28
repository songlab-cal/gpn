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
]


setup(
    name='gpn',
    version='0.1',
    description='gpn',
    url='http://github.com/songlab-cal/gpn',
    author='Gonzalo Benegas',
    author_email='gbenegas@berkeley.edu',
    license='MIT',
    packages=['gpn'],
    zip_safe=False,
    install_requires=install_requires,
)
