"""Tests for the com2ann.py script in the Tools/parser directory."""

import unittest
from com2ann import com2ann, TYPE_COM
import re
from textwrap import dedent


class BaseTestCase(unittest.TestCase):

    def check(self, code, expected, n=False, e=False, w=0):
        self.assertEqual(com2ann(dedent(code),
                         drop_none=n, drop_ellipsis=e, silent=True, wrap_sig=w),
                         dedent(expected) if expected is not None else None)


class AssignTestCase(BaseTestCase):
    def test_basics(self):
        self.check("z = 5", "z = 5")
        self.check("z: int = 5", "z: int = 5")
        self.check("z = 5 # type: int", "z: int = 5")
        self.check("z = 5 # type: int # comment",
                   "z: int = 5 # comment")

    def test_type_ignore(self):
        self.check("foobar = foo_baz() # type: ignore",
                   "foobar = foo_baz() # type: ignore")
        self.check("a = 42 #type: ignore #comment",
                   "a = 42 #type: ignore #comment")
        self.check("foobar = None  # type: int  # type: ignore",
                   "foobar: int  # type: ignore", True, False)

    def test_complete_tuple(self):
        self.check("t = 1, 2, 3 # type: Tuple[int, ...]",
                   "t: Tuple[int, ...] = (1, 2, 3)")
        self.check("t = 1, # type: Tuple[int]",
                   "t: Tuple[int] = (1,)")
        self.check("t = (1, 2, 3) # type: Tuple[int, ...]",
                   "t: Tuple[int, ...] = (1, 2, 3)")

    def test_drop_None(self):
        self.check("x = None # type: int",
                   "x: int", True)
        self.check("x = None # type: int # another",
                   "x: int # another", True)
        self.check("x = None # type: int # None",
                   "x: int # None", True)

    def test_drop_Ellipsis(self):
        self.check("x = ... # type: int",
                   "x: int", False, True)
        self.check("x = ... # type: int # another",
                   "x: int # another", False, True)
        self.check("x = ... # type: int # ...",
                   "x: int # ...", False, True)

    def test_newline(self):
        self.check("z = 5 # type: int\r\n", "z: int = 5\r\n")
        self.check("z = 5 # type: int # comment\x85",
                   "z: int = 5 # comment\x85")

    def test_wrong(self):
        self.check("#type : str", "#type : str")
        self.check("x==y #type: bool", None)  # this is syntax error
        self.check("x==y ##type: bool", "x==y ##type: bool")  # this is OK

    def test_pattern(self):
        for line in ["#type: int", "  # type:  str[:] # com"]:
            self.assertTrue(re.search(TYPE_COM, line))
        for line in ["", "#", "# comment", "#type", "type int:"]:
            self.assertFalse(re.search(TYPE_COM, line))

    def test_uneven_spacing(self):
        self.check('x = 5   #type: int # this one is OK',
                   'x: int = 5 # this one is OK')

    def test_coding_kept(self):
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

    def test_multi_line_tuple_value(self):
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

    def test_complex_targets(self):
        self.check("x = y = z = 1 # type: int",
                   "x = y = z = 1 # type: int")
        self.check("x, y, z = [], [], []  # type: (List[int], List[int], List[str])",
                   "x, y, z = [], [], []  # type: (List[int], List[int], List[str])")
        self.check("self.x = None  # type: int  # type: ignore",
                   "self.x: int  # type: ignore",
                   True, False)
        self.check("self.x[0] = []  # type: int  # type: ignore",
                   "self.x[0]: int = []  # type: ignore")

    def test_multi_line_assign(self):
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

    def test_parenthesized_lhs(self):
        self.check(
            """
            (C.x[1]) = \\
                42 == 5# type: bool
            """,
            """
            (C.x[1]): bool = \\
                42 == 5
            """)

    def test_literal_types(self):
        self.check("x = None  # type: Optional[Literal['#']]",
                   "x: Optional[Literal['#']] = None")


class FunctionTestCase(BaseTestCase):
    def test_single(self):
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

    def test_complex_kinds(self):
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
            """)
        self.check(
            """
            def embezzle(account, funds=MANY, *fake_receipts, stuff, other=None, **kwarg):  # type: ignore
                # type: (str, int, *str, Any, Optional[Any], Any) -> None
                pass
            """,
            """
            def embezzle(account: str, funds: int = MANY, *fake_receipts: str, stuff: Any, other: Optional[Any] = None, **kwarg: Any) -> None:  # type: ignore
                pass
            """)

    def test_self_argument(self):
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

    def test_combined_annotations_single(self):
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

    def test_combined_annotations_multi(self):
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

    def test_literal_type(self):
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

    def test_wrap_lines(self):
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


class ForAndWithTestCase(BaseTestCase):
    def test_with(self):
        # TODO: support this.
        self.check(
            """
            with foo(x==1) as f: #type: str
                print(f)
            """,
            """
            with foo(x==1) as f: #type: str
                print(f)
            """)

    def test_for(self):
        # TODO: support this.
        self.check(
            """
            for i, j in my_inter(x=1): # type: (int, int)  # type: ignore
                i + j
            """,
            """
            for i, j in my_inter(x=1): # type: (int, int)  # type: ignore
                i + j
            """)


if __name__ == '__main__':
    unittest.main()
