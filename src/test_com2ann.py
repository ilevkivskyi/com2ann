"""Tests for the com2ann.py script in the Tools/parser directory."""

import unittest
from com2ann import com2ann, TYPE_COM
import re
import sys
from textwrap import dedent


class BaseTestCase(unittest.TestCase):

    def check(self, code, expected, n=False, e=False):
        self.assertEqual(com2ann(dedent(code),
                         drop_none=n, drop_ellipsis=e, silent=True),
                         dedent(expected))


class SimpleTestCase(BaseTestCase):
    # Tests for basic conversions

    def test_basics(self):
        self.check("z = 5", "z = 5")
        self.check("z: int = 5", "z: int = 5")
        self.check("z = 5 # type: int", "z: int = 5")
        self.check("z = 5 # type: int # comment",
                   "z: int = 5 # comment")

    def test_type_ignore(self):
        self.check("foobar = foo_baz() #type: ignore",
                   "foobar = foo_baz() #type: ignore")
        self.check("a = 42 #type: ignore #comment",
                   "a = 42 #type: ignore #comment")

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
        self.check('x = 5 #type    : int # this one is OK',
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


if __name__ == '__main__':
    unittest.main()
