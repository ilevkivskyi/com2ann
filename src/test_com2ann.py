"""Tests for the com2ann.py script in the Tools/parser directory."""

import unittest
from com2ann import com2ann, TYPE_COM
import re
from textwrap import dedent
from typing import Optional, List


class BaseTestCase(unittest.TestCase):

    def check(self, code: str, expected: Optional[str],
              n: bool = False, e: bool = False,
              w: int = 0, i: bool = False) -> None:
        result = com2ann(dedent(code),
                         drop_none=n, drop_ellipsis=e, silent=True,
                         wrap_sig=w, add_future_imports=i)
        if expected is None:
            self.assertIs(result, None)
        else:
            assert result is not None
            new_code, _ = result
            self.assertEqual(new_code, dedent(expected))


class AssignTestCase(BaseTestCase):
    def test_basics(self) -> None:
        self.check("z = 5", "z = 5")
        self.check("z: int = 5", "z: int = 5")
        self.check("z = 5 # type: int", "z: int = 5")
        self.check("z = 5 # type: int # comment",
                   "z: int = 5 # comment")

    def test_type_ignore(self) -> None:
        self.check("foobar = foo_baz() # type: ignore",
                   "foobar = foo_baz() # type: ignore")
        self.check("a = 42 #type: ignore #comment",
                   "a = 42 #type: ignore #comment")
        self.check("foobar = None  # type: int  # type: ignore",
                   "foobar: int  # type: ignore", True, False)

    def test_complete_tuple(self) -> None:
        self.check("t = 1, 2, 3 # type: Tuple[int, ...]",
                   "t: Tuple[int, ...] = (1, 2, 3)")
        self.check("t = 1, # type: Tuple[int]",
                   "t: Tuple[int] = (1,)")
        self.check("t = (1, 2, 3) # type: Tuple[int, ...]",
                   "t: Tuple[int, ...] = (1, 2, 3)")

    def test_drop_None(self) -> None:
        self.check("x = None # type: int",
                   "x: int", True)
        self.check("x = None # type: int # another",
                   "x: int # another", True)
        self.check("x = None # type: int # None",
                   "x: int # None", True)

    def test_drop_Ellipsis(self) -> None:
        self.check("x = ... # type: int",
                   "x: int", False, True)
        self.check("x = ... # type: int # another",
                   "x: int # another", False, True)
        self.check("x = ... # type: int # ...",
                   "x: int # ...", False, True)

    def test_newline(self) -> None:
        self.check("z = 5 # type: int\r\n", "z: int = 5\r\n")
        self.check("z = 5 # type: int # comment\x85",
                   "z: int = 5 # comment\x85")

    def test_wrong(self) -> None:
        self.check("#type : str", "#type : str")
        self.check("x==y #type: bool", None)  # this is syntax error
        self.check("x==y ##type: bool", "x==y ##type: bool")  # this is OK

    def test_pattern(self) -> None:
        for line in ["#type: int", "  # type:  str[:] # com"]:
            self.assertTrue(re.search(TYPE_COM, line))
        for line in ["", "#", "# comment", "#type", "type int:"]:
            self.assertFalse(re.search(TYPE_COM, line))

    def test_uneven_spacing(self) -> None:
        self.check('x = 5   #type: int # this one is OK',
                   'x: int = 5 # this one is OK')

    def test_coding_kept(self) -> None:
        self.check(
            """
            # -*- coding: utf-8 -*- # this should not be spoiled
            '''
            Docstring here
            '''

            import testmod
            from typing import Optional

            coding = None  # type: Optional[str]
            """,
            """
            # -*- coding: utf-8 -*- # this should not be spoiled
            '''
            Docstring here
            '''

            import testmod
            from typing import Optional

            coding: Optional[str] = None
            """)

    def test_multi_line_tuple_value(self) -> None:
        self.check(
            """
            ttt \\
                 = \\
                   1.0, \\
                   2.0, \\
                   3.0, #type: Tuple[float, float, float]
            """,
            """
            ttt: Tuple[float, float, float] \\
                 = \\
                   (1.0, \\
                    2.0, \\
                    3.0,)
            """)

    def test_complex_targets(self) -> None:
        self.check("x = y = z = 1 # type: int",
                   "x = y = z = 1 # type: int")
        self.check("x, y, z = [], [], []  # type: (List[int], List[int], List[str])",
                   "x, y, z = [], [], []  # type: (List[int], List[int], List[str])")
        self.check("self.x = None  # type: int  # type: ignore",
                   "self.x: int  # type: ignore",
                   True, False)
        self.check("self.x[0] = []  # type: int  # type: ignore",
                   "self.x[0]: int = []  # type: ignore")

    def test_multi_line_assign(self) -> None:
        self.check(
            """
            class C:

                l[f(x
                    =1)] = [

                     g(y), # type: ignore
                     2,
                     ]  # type: List[int]
            """,
            """
            class C:

                l[f(x
                    =1)]: List[int] = [

                     g(y), # type: ignore
                     2,
                     ]
            """)

    def test_parenthesized_lhs(self) -> None:
        self.check(
            """
            (C.x[1]) = \\
                42 == 5# type: bool
            """,
            """
            (C.x[1]): bool = \\
                42 == 5
            """)

    def test_literal_types(self) -> None:
        self.check("x = None  # type: Optional[Literal['#']]",
                   "x: Optional[Literal['#']] = None")

    def test_comment_on_separate_line(self) -> None:
        self.check(
            """
            bar = {} \\
                # type: SuperLongType[WithArgs]
            """,
            """
            bar: SuperLongType[WithArgs] = {}
            """)
        self.check(
            """
            bar = {} \\
                # type: SuperLongType[WithArgs]  # noqa
            """,
            """
            bar: SuperLongType[WithArgs] = {} \\
                # noqa
            """)
        self.check(
            """
            bar = None \\
                # type: SuperLongType[WithArgs]
            """,
            """
            bar: SuperLongType[WithArgs]
            """, n=True)

    def test_continuation_using_parens(self) -> None:
        self.check(
            """
            X = (
                {one}
                | {other}
            )  # type: Final  # another option
            """,
            """
            X: Final = (
                {one}
                | {other}
            )  # another option
            """)
        self.check(
            """
            X = (  # type: ignore
                {one}
                | {other}
            )  # type: Final
            """,
            """
            X: Final = (  # type: ignore
                {one}
                | {other}
            )
            """)
        self.check(
            """
            foo = object()

            bar = (
                # Comment which explains why this ignored
                foo.quox   # type: ignore[attribute]
            )  # type: Mapping[str, Distribution]
            """,
            """
            foo = object()

            bar: Mapping[str, Distribution] = (
                # Comment which explains why this ignored
                foo.quox   # type: ignore[attribute]
            )
            """)

    def test_with_for(self) -> None:
        self.check(
            """
            for i in range(test):  # type: float
                with open('/some/file'):
                    def f():
                        # type: () -> None
                        x = []  # type: List[int]  # unused
            """,
            """
            for i in range(test):  # type: float
                with open('/some/file'):
                    def f() -> None:
                        x: List[int] = []  # unused
            """)


class FunctionTestCase(BaseTestCase):
    def test_single(self) -> None:
        self.check(
            """
            def add(a, b):  # type: (int, int) -> int
                '''# type: yes'''
            """,
            """
            def add(a: int, b: int) -> int:
                '''# type: yes'''
            """)
        self.check(
            """
            def add(a, b):  # type: (int, int) -> int  # extra comment
                pass
            """,
            """
            def add(a: int, b: int) -> int:  # extra comment
                pass
            """)

    def test_async_single(self) -> None:
        self.check(
            """
            async def add(a, b):  # type: (int, int) -> int
                '''# type: yes'''
            """,
            """
            async def add(a: int, b: int) -> int:
                '''# type: yes'''
            """)
        self.check(
            """
            async def add(a, b):  # type: (int, int) -> int  # extra comment
                pass
            """,
            """
            async def add(a: int, b: int) -> int:  # extra comment
                pass
            """)

    def test_complex_kinds(self) -> None:
        self.check(
            """
            def embezzle(account, funds=MANY, *fake_receipts, stuff, other=None, **kwarg):
                # type: (str, int, *str, Any, Optional[Any], Any) -> None  # note: vararg and kwarg
                pass
            """,
            """
            def embezzle(account: str, funds: int = MANY, *fake_receipts: str, stuff: Any, other: Optional[Any] = None, **kwarg: Any) -> None:
                # note: vararg and kwarg
                pass
            """)  # noqa
        self.check(
            """
            def embezzle(account, funds=MANY, *fake_receipts, stuff, other=None, **kwarg):  # type: ignore
                # type: (str, int, *str, Any, Optional[Any], Any) -> None
                pass
            """,
            """
            def embezzle(account: str, funds: int = MANY, *fake_receipts: str, stuff: Any, other: Optional[Any] = None, **kwarg: Any) -> None:  # type: ignore
                pass
            """)  # noqa

    def test_self_argument(self) -> None:
        self.check(
            """
            def load_cache(self):
                # type: () -> bool
                pass
            """,
            """
            def load_cache(self) -> bool:
                pass
            """)

    def test_combined_annotations_single(self) -> None:
        self.check(
            """
            def send_email(address, sender, cc, bcc, subject, body):
                # type: (...) -> bool
                pass
            """,
            """
            def send_email(address, sender, cc, bcc, subject, body) -> bool:
                pass
            """)
        # TODO: should we move an ignore on its own line somewhere else?
        self.check(
            """
            def send_email(address, sender, cc, bcc, subject, body):
                # type: (...) -> BadType  # type: ignore
                pass
            """,
            """
            def send_email(address, sender, cc, bcc, subject, body) -> BadType:
                # type: ignore
                pass
            """)
        self.check(
            """
            def send_email(address, sender, cc, bcc, subject, body):  # type: ignore
                # type: (...) -> bool
                pass
            """,
            """
            def send_email(address, sender, cc, bcc, subject, body) -> bool:  # type: ignore
                pass
            """)

    def test_combined_annotations_multi(self) -> None:
        self.check(
            """
            def send_email(address,     # type: Union[str, List[str]]
               sender,      # type: str
               cc,          # type: Optional[List[str]]  # this is OK
               bcc,         # type: Optional[List[Bad]]  # type: ignore
               subject='',
               body=None,   # type: List[str]
               *args        # type: ignore
               ):
               # type: (...) -> bool
               pass
            """,
            """
            def send_email(address: Union[str, List[str]],
               sender: str,
               cc: Optional[List[str]],  # this is OK
               bcc: Optional[List[Bad]],  # type: ignore
               subject='',
               body: List[str] = None,
               *args        # type: ignore
               ) -> bool:
               pass
            """
        )

    def test_literal_type(self) -> None:
        self.check(
            """
            def force_hash(
                arg,  # type: Literal['#']
            ):
                # type: (...) -> Literal['#']
                pass
            """,
            """
            def force_hash(
                arg: Literal['#'],
            ) -> Literal['#']:
                pass
            """)

    def test_wrap_lines(self) -> None:
        self.check(
            """
            def embezzle(self, account, funds=MANY, *fake_receipts):
                # type: (str, int, *str) -> None  # some comment
                pass
            """,
            """
            def embezzle(self,
                         account: str,
                         funds: int = MANY,
                         *fake_receipts: str) -> None:
                # some comment
                pass
            """, False, False, 10)
        self.check(
            """
            def embezzle(self, account, funds=MANY, *fake_receipts):  # type: ignore
                # type: (str, int, *str) -> None
                pass
            """,
            """
            def embezzle(self,  # type: ignore
                         account: str,
                         funds: int = MANY,
                         *fake_receipts: str) -> None:
                pass
            """, False, False, 10)
        self.check(
            """
            def embezzle(self, account, funds=MANY, *fake_receipts):
                # type: (str, int, *str) -> Dict[str, Dict[str, int]]
                pass
            """,
            """
            def embezzle(self,
                         account: str,
                         funds: int = MANY,
                         *fake_receipts: str) -> Dict[str, Dict[str, int]]:
                pass
            """, False, False, 10)

    def test_wrap_lines_error_code(self) -> None:
        self.check(
            """
            def embezzle(self, account, funds=MANY, *fake_receipts):  # type: ignore[override]
                # type: (str, int, *str) -> None
                pass
            """,
            """
            def embezzle(self,  # type: ignore[override]
                         account: str,
                         funds: int = MANY,
                         *fake_receipts: str) -> None:
                pass
            """, False, False, 10)

    def test_decorator_body(self) -> None:
        self.check(
            """
            def outer(self):  # a method
                # type: () -> None
                @deco()
                def inner():
                    # type: () -> None
                    pass
            """,
            """
            def outer(self) -> None:  # a method
                @deco()
                def inner() -> None:
                    pass
            """)
        self.check(
            """
            def func(
                x,  # type: int
                *other,  # type: Any
            ):
                # type: () -> None
                @dataclass
                class C:
                    x = None  # type: int
            """,
            """
            def func(
                x: int,
                *other: Any,
            ) -> None:
                @dataclass
                class C:
                    x: int
            """, n=True)

    def test_keyword_only_args(self) -> None:
        self.check(
            """
            def func(self,
                *,
                account,
                callback,  # type: Callable[[], None]
                start=0,  # type: int
                order,  # type: bool
                ):
                # type: (...) -> None
                ...
            """,
            """
            def func(self,
                *,
                account,
                callback: Callable[[], None],
                start: int = 0,
                order: bool,
                ) -> None:
                ...
            """)

    def test_next_line_comment(self) -> None:
        self.check(
            """
            def __init__(
                self,
                short,                # type: Short
                long_argument,
                # type: LongType[int, str]
                other,                # type: Other
            ):
                # type: (...) -> None
                '''
                Some function.
                '''
            """,
            """
            def __init__(
                self,
                short: Short,
                long_argument: LongType[int, str],
                other: Other,
            ) -> None:
                '''
                Some function.
                '''
            """)


class LineReportingTestCase(BaseTestCase):
    def compare(self, code: str, success: List[int], fail: List[int]) -> None:
        result = com2ann(dedent(code), silent=True)
        assert result is not None
        _, data = result
        self.assertEqual(data.success, success)
        self.assertEqual(data.fail, fail)

    def test_simple_assign(self) -> None:
        self.compare(
            """
            x = None  # type: Optional[str]
            """,
            [2], [])

    def test_simple_function(self) -> None:
        self.compare(
            """
            def func(arg):
                # type: (int) -> int
                pass
            """,
            [2], [])

    def test_unsupported_assigns(self) -> None:
        self.compare(
            """
            x, y = None, None  # type: (int, int)
            x = None  # type: Optional[str]
            x = y = []  # type: List[int]
            """,
            [3], [2, 4])

    def test_invalid_function_comments(self) -> None:
        self.compare(
            """
            def func(arg):
                # type: bad
                pass
            def func(arg):
                # type: bad -> bad
                pass
            """,
            [], [2, 5])

    def test_confusing_function_comments(self) -> None:
        self.compare(
            """
            def func1(
                arg  # type: int
            ):
                # type: (str) -> int
                pass
            def func2(arg1, arg2, arg3):
                # type: (int) -> int
                pass
            """,
            [], [2, 7])

    def test_unsupported_statements(self) -> None:
        self.compare(
            """
            with foo(x==1) as f: # type: str
                print(f)
            with foo(x==1) as f:
                print(f)
            x = None  # type: Optional[str]
            for i, j in my_inter(x=1):
                i + j
            for i, j in my_inter(x=1): # type: (int, int)  # type: ignore
                i + j
            """,
            [6], [2, 9])


class ForAndWithTestCase(BaseTestCase):
    def test_with(self) -> None:
        self.check(
            """
            with foo(x==1) as f: #type: str
                print(f)
            """,
            """
            with foo(x==1) as f: #type: str
                print(f)
            """)

    def test_for(self) -> None:
        self.check(
            """
            for i, j in my_inter(x=1): # type: (int, int)  # type: ignore
                i + j
            """,
            """
            for i, j in my_inter(x=1): # type: (int, int)  # type: ignore
                i + j
            """)

    def test_async_with(self) -> None:
        self.check(
            """
            async with foo(x==1) as f: #type: str
                print(f)
            """,
            """
            async with foo(x==1) as f: #type: str
                print(f)
            """)

    def test_async_for(self) -> None:
        self.check(
            """
            async for i, j in my_inter(x=1): # type: (int, int)  # type: ignore
                i + j
            """,
            """
            async for i, j in my_inter(x=1): # type: (int, int)  # type: ignore
                i + j
            """)


class FutureImportTestCase(BaseTestCase):
    def test_added_future_import(self) -> None:
        self.check(
            """
            # coding: utf-8

            x = None  # type: Optional[str]
            """,
            """
            # coding: utf-8

            from __future__ import annotations
            x: Optional[str] = None
            """, i=True)

    def test_not_added_future_import(self) -> None:
        self.check(
            """
            x = 1
            """,
            """
            x = 1
            """, i=True)
        self.check(
            """
            x, y = a, b  # type: Tuple[int, int]
            """,
            """
            x, y = a, b  # type: Tuple[int, int]
            """, i=True)
        self.check(
            """
            def foo(arg1, arg2):
                # type: (int, str) -> None
                pass
            x = False  # type: bool
            """,
            """
            def foo(arg1: int, arg2: str) -> None:
                pass
            x: bool = False
            """, i=True)


if __name__ == '__main__':
    unittest.main()
