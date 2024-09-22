from setuptools import setup, find_packages

setup(
    name="ml_microclimate",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'pytorch-lightning<=2.0.0',
        'transformers>=4.0.0',
        'albumentations>=0.5.0',
        'datasets>=1.6.0',
        'torchmetrics>=0.5.0',
        'evaluate>=0.1.0',
        'tqdm>=4.41.0',
        'scikit-learn',
        'albumentations',
        'transformers',
        'huggingface_hub'
    ],
    entry_points={
        'console_scripts': [
            #  CLI command 'ml'
            'ml = scripts.main:main',
        ],
    },
)
