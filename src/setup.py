from setuptools import setup

setup(
    name='cr_summ',
    version='0.0.3',
    packages=['cr_summ', 'cr_summ.modules'],
    package_data={'cr_summ.modules': ['src/modules/lib/*.so']},
    install_requires=[
        # 'datasets==2.16.1',
        # 'numpy==1.26.3',
        # 'pandas==2.1.4',
        # 'pyyaml==6.0.1',
        # 'torch==2.1.2',
        # 'tokenizers==0.15.0',
        # 'unsloth==2024.1'
    ],
    extras_requires=[
        'tensorboard==2.15',
        'tvm'
    ]
)
