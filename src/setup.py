from setuptools import setup

setup(
    name='CRD3_summarization',
    version='0.0.2',
    packages=['CRD3_summarization', 'CRD3_summarization.models', 'CRD3_summarization.datasets', 'CRD3_summarization.modules'],
    package_data={'CRD3_summarization.modules': ['src/modules/lib/*.so']},
#    install_requires=[
#        'numpy>=1.24',
#        'pandas>=1.5',
#        'pyyaml>=6.0',
#        'torch>=1.13',
#        'tokenizers>=0.13'
#    ],
    extras_requires=[
        'tensorboard>=2.12'
    ]
)
