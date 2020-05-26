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
        # type (str, **bool) -> str
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
