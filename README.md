com2ann
=======

[![Build Status](https://travis-ci.org/ilevkivskyi/com2ann.svg)](https://travis-ci.org/ilevkivskyi/com2ann)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

Tool for translation of type comments to type annotations in Python.

The tool requires Python 3.8 to run. But the supported target code version
is Python 3.4+ (can be specified with `--python-minor-version`).

Currently, the tool translates function and assignment type comments to
type annotations. For example this code:
```python
from typing import Optional, Final

MAX_LEVEL = 80  # type: Final

class Template:
    default = None  # type: Optional[str]

    def apply(self, value, **opts):
        # type: (str, **bool) -> str
        ...
```
will be translated to:
```python
from typing import Optional, Final

MAX_LEVEL: Final = 80

class Template:
    default: Optional[str] = None

    def apply(self, value: str, **opts: str) -> str:
        ...
```

The philosophy of the tool is to be minimally invasive, and preserve original
formatting as much as possible. This is why the tool doesn't use un-parse.

The only (optional) formatting code modification is wrapping long function
signatures. To specify the maximal length, use `--wrap-signatures MAX_LENGTH`.
The signatures are wrapped one argument per line (after each comma), for example:
```python
    def apply(self,
              value: str,
              **opts: str) -> str:
        ...
```

For working with stubs, there are two additional options for assignments:
`--drop-ellipsis` and `--drop-none`. They will result in omitting the redundant
right hand sides. For example, this:
```python
var = ...  # type: List[int]
class Test:
    attr = None  # type: str
```
will be translated with such options to:
```python
var: List[int]
class Test:
    attr: str
```
### Usage
$ `com2ann --help`
```
usage: com2ann [-h] [-o OUTFILE] [-s] [-n] [-e] [-i] [-w WRAP_SIGNATURES]
               [-v PYTHON_MINOR_VERSION]
               infile

Helper module to translate type comments to type annotations. The key idea of
this module is to perform the translation while preserving the original
formatting as much as possible. We try to be not opinionated about code
formatting and therefore work at the source code and tokenizer level instead
of modifying AST and using un-parse. We are especially careful about
assignment statements, and keep the placement of additional (non-type)
comments. For function definitions, we might introduce some formatting
modifications, if the original formatting was too tricky.

positional arguments:
  infile                input file or directory for translation, must contain
                        no syntax errors; if --outfile is not given,
                        translation is made *in place*

optional arguments:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        output file or directory, will be overwritten if
                        exists, defaults to input file or directory
  -s, --silent          do not print summary for line numbers of translated
                        and rejected comments
  -n, --drop-none       drop any None as assignment value during translation
                        if it is annotated by a type comment
  -e, --drop-ellipsis   drop any Ellipsis (...) as assignment value during
                        translation if it is annotated by a type comment
  -i, --add-future-imports
                        add 'from __future__ import annotations' to any file
                        where type comments were successfully translated
  -w WRAP_SIGNATURES, --wrap-signatures WRAP_SIGNATURES
                        wrap function headers that are longer than given
                        length
  -v PYTHON_MINOR_VERSION, --python-minor-version PYTHON_MINOR_VERSION
                        Python 3 minor version to use to parse the files
```
