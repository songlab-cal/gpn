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
    name='GPN',
    version='0.1',
    description='GPN',
    url='http://github.com/gonzalobenegas/gpn',
    author='Gonzalo Benegas',
    author_email='gbenegas@berkeley.edu',
    license='MIT',
    packages=['gpn'],
    zip_safe=False,
    install_requires=install_requires
)
