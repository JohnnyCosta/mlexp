from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='mlexp',
    version='0.1',
    description='keras on gcloud ml-engine',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    url='https://github.com//mlexp',
    author='',
    # author_email='@gmail.com',
    license='MIT',
    packages=['mlexp'],
    install_requires=[
        'keras',
        'h5py',
        'numpy',
        'pillow'
    ],
    include_package_data=True,
    zip_safe=False
)
