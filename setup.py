from setuptools import setup

setup(
    name='tkitKeyBertBackend',
    version='0.0.1',
    packages=['tkitKeyBertBackend'],
    url='https://terrychan.org/2022/03/keybert%e5%81%9a%e4%b8%ad%e6%96%87%e6%96%87%e6%9c%ac%e5%85%b3%e9%94%ae%e8%af%8d%e6%8f%90%e5%8f%96/',
    license='GNU GENERAL PUBLIC LICENSE  Version 3, 29 June 2007',
    author='terry',
    author_email='napoler2008@gmail.com',
    description='这里提供keybert引入huggingface transformers作为后端，可以方便处理中文',
    install_requires=["transformers>=4.17.0","keybert>=0.5.0"]
)
