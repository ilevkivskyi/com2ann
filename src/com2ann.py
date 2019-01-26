"""Helper module to translate type comments to type annotations.

The key idea of this module is to perform the translation while preserving
the original formatting as much as possible. We try to be not opinionated
about code formatting and therefore work at the source code and tokenizer level
instead of modifying AST and using un-parse.

We are especially careful about assignment statements, and keep the placement
of additional (non-type) comments. For function definitions, we might introduce
some formatting modifications, if the original formatting was too tricky.
"""
import re
import os
import ast
from ast import Module
import argparse
import tokenize
from tokenize import TokenInfo
from collections import defaultdict
from textwrap import dedent
from io import BytesIO

from typing import List, DefaultDict, Tuple, Optional

__all__ = ['com2ann', 'TYPE_COM']

TYPE_COM = re.compile(r'\s*#\s*type\s*:.*$', flags=re.DOTALL)
TRAIL_OR_COM = re.compile(r'\s*$|\s*#.*$', flags=re.DOTALL)


class Data:
    """Internal class describing global data on file."""
    def __init__(self, lines: List[str], tokens: List[TokenInfo]):
        # Source code lines.
        self.lines = lines
        # Tokens for the source code.
        self.tokens = tokens
        # Map line number to token numbers. For example {1: [0, 1, 2, 3], 2: [4, 5]}
        # means that first to fourth tokens are on the first line.
        token_tab: DefaultDict[int, List[int]] = defaultdict(list)
        for i, tok in enumerate(tokens):
            token_tab[tok.start[0]].append(i)
        self.token_tab = token_tab
        self.success: List[int] = []  # list of lines where type comments where processed
        self.fail: List[int] = []  # list of lines where type comments where rejected


def skip_blank(d: Data, line: int) -> int:
    """Return first non-blank line number after `line`."""
    while not d.lines[line].strip():
        line += 1
    return line


def find_start(d: Data, line_com: int) -> int:
    """Find line where first char of the assignment target appears.

    `line_com` is the line where type comment was found.
    """
    i = d.token_tab[line_com + 1][-2]  # index of type comment token in tokens list
    # First climb back to end of previous statement.
    while ((d.tokens[i].exact_type != tokenize.NEWLINE) and
           (d.tokens[i].exact_type != tokenize.ENCODING)):
        i -= 1
    lno = d.tokens[i].start[0]
    return skip_blank(d, lno)


def check_target(stmt: Module) -> bool:
    """Check if the statement is suitable for annotation.

    Type comments can placed on with and for statements, but
    annotation can be placed only on an simple assignment with a single target.
    """
    if len(stmt.body):
        assign = stmt.body[0]
    else:
        return False
    if isinstance(assign, ast.Assign) and len(assign.targets) == 1:
        target = assign.targets[0]
    else:
        return False
    if (
        isinstance(target, ast.Name) or isinstance(target, ast.Attribute) or
        isinstance(target, ast.Subscript)
    ):
        return True
    return False


def find_eq(d: Data, line_start: int) -> Tuple[int, int]:
    """Find equal sign position starting from `line_start`.

    We need to be careful about not taking first assignment in d[f(x=1)] = 5.
    """
    col = pars = 0
    line = line_start
    while d.lines[line][col] != '=' or pars != 0:
        ch = d.lines[line][col]
        if ch in '([{':
            pars += 1
        elif ch in ')]}':
            pars -= 1
        # A comment or blank line in the middle of assignment statement -- skip it.
        if ch == '#' or col == len(d.lines[line]) - 1:
            line = skip_blank(d, line + 1)
            col = 0
        else:
            col += 1
    return line, col


def find_val(d: Data, pos_eq: Tuple[int, int]) -> Tuple[int, int]:
    """Find position of first character of the assignment r.h.s.

    `pos_eq` is the position of equality sign in the assignment.
    """
    line, col = pos_eq
    # Walk forward from equality sign.
    while d.lines[line][col].isspace() or d.lines[line][col] in '=\\':
        if col == len(d.lines[line]) - 1:
            line += 1
            col = 0
        else:
            col += 1
    return line, col


def find_target(d: Data, pos_eq: Tuple[int, int]) -> Tuple[int, int]:
    """Find position of last character of the target (annotation goes here)."""
    line, col = pos_eq
    # Walk backward from the equality sign.
    while d.lines[line][col].isspace() or d.lines[line][col] in '=\\':
        if col == 0:
            line -= 1
            col = len(d.lines[line]) - 1
        else:
            col -= 1
    return line, col + 1


def trim(new_lines: List[str], string: str,
         line_target: int, pos_eq: Tuple[int, int],
         line_com: int, col_com: int) -> None:
    """Remove None or Ellipsis from assignment value.

    Also remove parentheses if one has (None), (...) etc.
    This modifies the `new_lines` in place.

    Arguments:
    * string: 'None' or '...'
    * line_target: line where last char of target is located
    * pos_eq: position of the equality sign
    * line_com, col_com: position of the type comment
    """
    def no_pars(s: str) -> str:
        return s.replace('(', '').replace(')', '')
    line_eq, col_eq = pos_eq

    sub_line = new_lines[line_eq][:col_eq]
    if line_eq == line_target:
        sub_line = sub_line.rstrip()
        replacement = new_lines[line_eq][col_com:]
    else:
        replacement = new_lines[line_eq][col_eq + 1:]

    new_lines[line_eq] = sub_line + replacement

    # Strip all parentheses between equality sign an type comment.
    for line in range(line_eq + 1, line_com):
        new_lines[line] = no_pars(new_lines[line])

    if line_com != line_eq:
        sub_line = no_pars(new_lines[line_com][:col_com]).replace(string, '')
        if not sub_line.isspace():
            sub_line = sub_line.rstrip()
        new_lines[line_com] = sub_line + new_lines[line_com][col_com:]


def com2ann_impl(d: Data, drop_none: bool, drop_ellipsis: bool) -> str:
    new_lines = d.lines[:]
    for line_com, line in enumerate(d.lines):
        match = re.search(TYPE_COM, line)
        if not match:
            continue

        # strip " #  type  :  annotation  \n" -> "annotation  \n"
        typ = match.group().lstrip()[1:].lstrip()[4:].lstrip()[1:].lstrip()
        sub_match = re.search(TRAIL_OR_COM, typ)
        sub_comment = ''
        if sub_match and sub_match.group():
            sub_comment = sub_match.group()
            typ = typ[:sub_match.start()]
        if typ == 'ignore':
            continue
        col_com = match.start()
        if not any(d.tokens[i].exact_type == tokenize.COMMENT
                   for i in d.token_tab[line_com + 1]):
            d.fail.append(line_com)
            continue  # type comment inside string
        line_start = find_start(d, line_com)
        stmt_str = dedent(''.join(d.lines[line_start:line_com + 1]))
        try:
            stmt = ast.parse(stmt_str)
        except SyntaxError:
            d.fail.append(line_com)
            continue  # for or with statements
        if not check_target(stmt):
            d.fail.append(line_com)
            continue

        d.success.append(line_com)
        val = stmt.body[0].value

        # writing output now
        pos_eq = find_eq(d, line_start)
        line_val, col_val = find_val(d, pos_eq)
        line_target, col_target = find_target(d, pos_eq)

        op_par = ''
        cl_par = ''
        if isinstance(val, ast.Tuple):
            if d.lines[line_val][col_val] != '(':
                op_par = '('
                cl_par = ')'
        # write the comment first
        new_lines[line_com] = d.lines[line_com][:col_com].rstrip() + cl_par + sub_comment
        col_com = len(d.lines[line_com][:col_com].rstrip())

        string = False
        if isinstance(val, ast.Tuple):
            # t = 1, 2 -> t = (1, 2); only latter is allowed with annotation
            free_place = int(new_lines[line_val][col_val - 2:col_val] == '  ')
            new_lines[line_val] = (new_lines[line_val][:col_val - free_place] +
                                   op_par + new_lines[line_val][col_val:])
        elif isinstance(val, ast.Ellipsis) and drop_ellipsis:
            string = '...'
        elif (isinstance(val, ast.NameConstant) and
              val.value is None and drop_none):
            string = 'None'
        if string:
            trim(new_lines, string, line_target, pos_eq, line_com, col_com)

        # finally write an annotation
        new_lines[line_target] = (new_lines[line_target][:col_target] +
                                  ': ' + typ + new_lines[line_target][col_target:])
    return ''.join(new_lines)


def com2ann(code: str, *, drop_none: bool = False, drop_ellipsis: bool = False,
            silent: bool = False) -> Optional[str]:
    """Translate type comments to type annotations in code.

    Take code as string and return this string where::

      variable = value  # type: annotation  # real comment

    is translated to::

      variable: annotation = value  # real comment

    For unsupported syntax cases, the type comments are
    left intact. If drop_None is True or if drop_Ellipsis
    is True translate correspondingly::

      variable = None  # type: annotation
      variable = ...  # type: annotation

    into::

      variable: annotation

    The tool tries to preserve code formatting as much as
    possible, but an exact translation is not guaranteed.
    A summary of translated comments id printed by default.
    """
    try:
        ast.parse(code)  # we want to work only with file without syntax errors
    except SyntaxError:
        return None
    lines = code.splitlines(keepends=True)
    rl = BytesIO(code.encode('utf-8')).readline
    tokens = list(tokenize.tokenize(rl))

    data = Data(lines, tokens)
    new_code = com2ann_impl(data, drop_none, drop_ellipsis)

    if not silent:
        if data.success:
            print('Comments translated on lines:',
                  ', '.join(str(lno + 1) for lno in data.success))
        if data.fail:
            print('Comments rejected on lines:',
                  ', '.join(str(lno + 1) for lno in data.fail))
        if not data.success and not data.fail:
            print('No type comments found')

    return new_code


def translate_file(infile: str, outfile: str,
                   drop_none: bool, drop_ellipsis: bool, silent: bool) -> None:
    try:
        opened = tokenize.open(infile)
    except SyntaxError:
        print("Cannot open", infile)
        return
    with opened as f:
        code = f.read()
        enc = f.encoding
    if not silent:
        print('File:', infile)
    new_code = com2ann(code, drop_none=drop_none,
                       drop_ellipsis=drop_ellipsis, silent=silent)
    if new_code is None:
        print("SyntaxError in", infile)
        return
    with open(outfile, 'wb') as f:
        f.write(new_code.encode(enc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--outfile",
                        help="output file, will be overwritten if exists,\n"
                             "defaults to input file")
    parser.add_argument("infile",
                        help="input file or directory for translation, must\n"
                             "contain no syntax errors, for directory\n"
                             "the outfile is ignored and translation is\n"
                             "made in place")
    parser.add_argument("-s", "--silent",
                        help="Do not print summary for line numbers of\n"
                             "translated and rejected comments",
                        action="store_true")
    parser.add_argument("-n", "--drop-none",
                        help="drop any None as assignment value during\n"
                        "translation if it is annotated by a type comment",
                        action="store_true")
    parser.add_argument("-e", "--drop-ellipsis",
                        help="drop any Ellipsis (...) as assignment value during\n"
                        "translation if it is annotated by a type comment",
                        action="store_true")
    args = parser.parse_args()
    if args.outfile is None:
        args.outfile = args.infile

    if os.path.isfile(args.infile):
        translate_file(args.infile, args.outfile,
                       args.drop_none, args.drop_ellipsis, args.silent)
    else:
        for root, _, files in os.walk(args.infile):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == '.py' or ext == '.pyi':
                    file_name = os.path.join(root, file)
                    translate_file(file_name, file_name,
                                   args.drop_none, args.drop_ellipsis,
                                   args.silent)
