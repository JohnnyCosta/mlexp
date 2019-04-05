# mlexp

## Requirements

- Python 3.5.2+

## Install

```
pip install git@github.com:/mlexp.git#egg=mlexp
```

## Develop

This package comes with a setup.sh script which swiftly
creates a virtualenv and installs dependencies from requirements.txt
without the hassle of virtualenv wrapper:

```
. ./setup.sh -p python3.5.2
```

## Test

```
py.test -v -s --cov-report term-missing --cov=mlexp -r w tests
```

## License

[MIT](LICENSE) 2019 
