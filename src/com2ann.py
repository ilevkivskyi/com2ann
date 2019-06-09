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
import argparse
import tokenize
from tokenize import TokenInfo
from collections import defaultdict
from textwrap import dedent
from io import BytesIO
from dataclasses import dataclass

from typing import List, DefaultDict, Tuple, Optional, Union

__all__ = ['com2ann', 'TYPE_COM']

TYPE_COM = re.compile(r'\s*#\s*type\s*:(.*)$', flags=re.DOTALL)
TRAIL_OR_COM = re.compile(r'\s*$|\s*#.*$', flags=re.DOTALL)


@dataclass
class AssignData:
    type_comment: str

    lvalue_end_line: int
    lvalue_end_offset: int

    rvalue_start_line: int
    rvalue_start_offset: int
    rvalue_end_line: int
    rvalue_end_offset: int

    tuple_rvalue: bool = False
    none_rvalue: bool = False
    ellipsis_rvalue: bool = False


@dataclass
class ArgComment:
    type_comment: str

    arg_line: int
    arg_end_offset: int

    has_default: bool = False


@dataclass
class FunctionData:
    arg_types: List[ArgComment]
    ret_type: Optional[str]

    header_start_line: int
    body_first_line: int


class FileData:
    """Internal class describing global data on file."""
    def __init__(self, lines: List[str], tokens: List[TokenInfo], tree: ast.AST) -> None:
        # Source code lines.
        self.lines = lines
        # Tokens for the source code.
        self.tokens = tokens
        # Parsed tree (with type_comments = True).
        self.tree = tree

        # Map line number to token numbers. For example {1: [0, 1, 2, 3], 2: [4, 5]}
        # means that first to fourth tokens are on the first line.
        token_tab: DefaultDict[int, List[int]] = defaultdict(list)
        for i, tok in enumerate(tokens):
            token_tab[tok.start[0]].append(i)
        self.token_tab = token_tab

        # Basic translation logging.
        self.success: List[int] = []  # list of lines where type comments where processed
        self.fail: List[int] = []  # list of lines where type comments where rejected


class TypeCommentCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.found: List[Union[AssignData, FunctionData]] = []

    def visit_Assign(self, s: ast.Assign) -> None:
        if s.type_comment:
            if not check_target(s):
                return
            target = s.targets[0]
            tuple_rvalue = isinstance(s.value, ast.Tuple)
            none_rvalue = isinstance(s.value, ast.Constant) and s.value.value is None
            ellipsis_rvalue = isinstance(s.value, ast.Constant) and s.value.value is Ellipsis
            found = AssignData(s.type_comment,
                               target.end_lineno, target.end_col_offset,
                               s.value.lineno, s.value.col_offset,
                               s.value.end_lineno, s.value.end_col_offset,
                               tuple_rvalue, none_rvalue, ellipsis_rvalue)
            self.found.append(found)

    def visit_FunctionDef(self, fdef: ast.FunctionDef) -> None:
        if (fdef.type_comment or
                any(a.type_comment for a in fdef.args.args) or
                any(a.type_comment for a in fdef.args.kwonlyargs) or
                fdef.args.vararg and fdef.vararg.type_comment or
                fdef.args.kwarg and fdef.args.kwarg.type_comment):
            num_non_defs = len(fdef.args.args) - len(fdef.args.defaults)
            num_kw_non_defs = len(fdef.args.kwonlyargs) - len([d for d in fdef.args.kw_defaults if d is not None])

            args = self.process_per_arg_comments(fdef, num_non_defs, num_kw_non_defs)

            if fdef.type_comment:
                f_args, ret = split_function_comment(fdef.type_comment)
            else:
                f_args = [], ret = None

            if args and f_args:
                # TODO: handle gracefully.
                raise Exception('Bad')

            if args:
                self.found.append(FunctionData(args, ret, fdef.lineno, fdef.body[0].lineno))
            elif not f_args:
                self.found.append(FunctionData([], ret, fdef.lineno, fdef.body[0].lineno))
            else:
                args = self.process_function_comment(fdef, f_args, num_non_defs, num_kw_non_defs)
                self.found.append(FunctionData(args, ret, fdef.lineno, fdef.body[0].lineno))
        self.generic_visit(fdef)

    def process_per_arg_comments(self, fdef: ast.FunctionDef,
                                 num_non_defs: int, num_kw_non_defs: int) -> List[ArgComment]:
        args: List[ArgComment] = []

        for i, a in enumerate(fdef.args.args):
            if a.type_comment:
                args.append(ArgComment(a.type_comment,
                                       a.lineno, a.end_col_offset,
                                       i >= num_non_defs))
        if fdef.args.vararg and fdef.args.vararg.type_comment:
            args.append(ArgComment(fdef.args.vararg.type_comment,
                                   fdef.args.vararg.lineno, fdef.args.vararg.end_col_offset,
                                   False))

        for i, a in enumerate(fdef.args.kwonlyargs):
            if a.type_comment:
                args.append(ArgComment(a.type_comment,
                                       a.lineno, a.end_col_offset,
                                       i >= num_kw_non_defs))
        if fdef.args.kwarg and fdef.args.kwarg.type_comment:
            args.append(ArgComment(fdef.args.kwarg.type_comment,
                                   fdef.args.kwarg.lineno, fdef.args.kwarg.end_col_offset,
                                   False))
        return args

    def process_function_comment(self, fdef: ast.FunctionDef, f_args: List[str],
                                 num_non_defs: int, num_kw_non_defs: int) -> List[ArgComment]:
        args: List[ArgComment] = []

        tot_args = len(fdef.args.args) + len(fdef.args.kwonlyargs)
        if fdef.args.vararg:
            tot_args += 1
        if fdef.args.kwarg:
            tot_args += 1

        if len(f_args) not in (tot_args, tot_args - 1):
            # TODO: handle gracefully.
            raise Exception('Bad')

        if len(f_args) == tot_args - 1:
            iter_args = fdef.args.args[1:]
        else:
            iter_args = fdef.args.args.copy()

        if fdef.args.vararg:
            iter_args.append(fdef.args.vararg)
        iter_args.extend(fdef.args.kwonlyargs)
        if fdef.args.kwarg:
            iter_args.append(fdef.args.kwarg)

        for typ, a in zip(f_args, iter_args):
            has_default = False
            if a in fdef.args.args and fdef.args.args.index(a) >= num_non_defs:
                has_default = True
            if a in fdef.args.kwonlyargs and fdef.args.kwonlyargs.index(a) >= num_kw_non_defs:
                has_default = True
            args.append(ArgComment(typ,
                                   a.lineno, a.end_col_offset,
                                   has_default))
        return args


# TODO: use tokenizer for split_function_comment() and split_sub_comment().


def split_sub_comment(comment: str) -> str:
    # TODO: take care of Literal['#'].
    return comment.split('#', maxsplit=1)[0].rstrip()


def split_function_comment(comment: str) -> Tuple[List[str], str]:
    # TODO: ()->int vs () -> int -- preserve spacing (maybe also # type:int vs # type: int)
    # TODO: fail gracefully on invalid types.

    typ = split_sub_comment(comment)
    assert '->' in typ, 'Invalid function type'
    arg_list, ret = typ.split('->')

    arg_list = arg_list.strip()
    ret = ret.strip()

    assert arg_list[0] == '(' and arg_list[-1] == ')'
    arg_list = arg_list[1:-1]

    args: List[str] = []

    next_arg = ''
    nested = 0
    for c in arg_list:
        if c in '([{':
            nested += 1
        if c in ')]}':
            nested -= 1
        if c == ',' and not nested:
            args.append(next_arg.strip())
            next_arg = ''
        else:
            next_arg += c

    if next_arg:
        args.append(next_arg.strip())

    return [a.lstrip('*') for a in args if a != '...'], ret


def strip_type_comment(line: str) -> str:
    match = re.search(TYPE_COM, line)
    assert match
    if match.group(1).lstrip().startswith('ignore'):
        # Keep # type: ignore[=code] comments.
        return line
    matched = line[match.start():]
    matched = matched.lstrip()[1:]

    rest = line[:match.start()]

    # TODO: take care of Literal['#'] also here.
    sub_comment = re.search(TRAIL_OR_COM, matched)
    assert sub_comment
    if rest:
        new_line = rest + matched[sub_comment.start():]
    else:
        # A type comment on line of its own.
        new_line = line[:line.index('#')] + matched[sub_comment.start():].lstrip(' \t')
    return new_line


def string_insert(line: str, extra: str, pos: int) -> str:
    return line[:pos] + extra + line[pos:]


def process_assign(comment: AssignData, data: FileData,
                   drop_none: bool, drop_ellipsis: bool) -> None:
    lines = data.lines
    lines[comment.rvalue_end_line - 1] = strip_type_comment(lines[comment.rvalue_end_line - 1])

    if comment.tuple_rvalue:
        # TODO: take care of (1, 2), (3, 4) with matching pars.
        if not (lines[comment.rvalue_start_line - 1][comment.rvalue_start_offset] == '(' and
                lines[comment.rvalue_end_line - 1][comment.rvalue_end_offset - 1] == ')'):
            # We need to wrap rvalue in parentheses before Python 3.8.
            end_line = lines[comment.rvalue_end_line - 1]
            lines[comment.rvalue_end_line - 1] = string_insert(end_line, ')', comment.rvalue_end_offset)

            start_line = lines[comment.rvalue_start_line - 1]
            lines[comment.rvalue_start_line - 1] = string_insert(start_line, '(', comment.rvalue_start_offset)

            if comment.rvalue_end_line > comment.rvalue_start_line:
                # Add a space to fix indentation after inserting paren.
                for i in range(comment.rvalue_end_line, comment.rvalue_start_line, -1):
                    if lines[i - 1].strip():
                        lines[i - 1] = ' ' + lines[i - 1]

    elif comment.none_rvalue and drop_none or comment.ellipsis_rvalue and drop_ellipsis:
        # TODO: more tricky (multi-line) cases.
        assert comment.lvalue_end_line == comment.rvalue_end_line
        line = lines[comment.lvalue_end_line - 1]
        lines[comment.lvalue_end_line - 1] = line[:comment.lvalue_end_offset] + line[comment.rvalue_end_offset:]

    lvalue_line = lines[comment.lvalue_end_line - 1]

    typ = split_sub_comment(comment.type_comment)
    lines[comment.lvalue_end_line - 1] = (lvalue_line[:comment.lvalue_end_offset] +
                                          ': ' + typ +
                                          lvalue_line[comment.lvalue_end_offset:])


def insert_arg_type(line: str, arg: ArgComment) -> str:
    typ = split_sub_comment(arg.type_comment)

    new_line = line[:arg.arg_end_offset] + ': ' + typ

    rest = line[arg.arg_end_offset:]
    if not arg.has_default:
        return new_line + rest

    # Here we are a bit opinionated about spacing (see PEP 8).
    rest = rest.lstrip()
    assert rest[0] == '=', (line, rest, arg)
    rest = rest[1:].lstrip()

    return new_line + ' = ' + rest


def process_func_def(func_type: FunctionData, data: FileData) -> None:
    lines = data.lines

    # Find column where _actual_ colon is located.
    ret_line = func_type.body_first_line - 1
    ret_line -= 1
    while not lines[ret_line].split('#')[0].strip():
        ret_line -= 1

    colon = None
    for i in reversed(data.token_tab[ret_line + 1]):
        if data.tokens[i].exact_type == tokenize.COLON:
            _, colon = data.tokens[i].start
            break
    assert colon is not None

    for i in range(func_type.body_first_line - 2, func_type.header_start_line - 2, -1):
        if re.search(TYPE_COM, lines[i]):
            lines[i] = strip_type_comment(lines[i])
            if not lines[i].strip():
                del lines[i]

    # Inserting return type is a bit dirty...
    if func_type.ret_type:
        right_par = lines[ret_line][:colon].rindex(')')
        lines[ret_line] = lines[ret_line][:right_par + 1] + ' -> ' + func_type.ret_type + lines[ret_line][colon:]

    for arg in reversed(func_type.arg_types):
        lines[arg.arg_line - 1] = insert_arg_type(lines[arg.arg_line - 1], arg)


def com2ann_impl(data: FileData, drop_none: bool, drop_ellipsis: bool) -> str:
    finder = TypeCommentCollector()
    finder.visit(data.tree)

    found = list(reversed(finder.found))

    for item in found:
        if isinstance(item, AssignData):
            process_assign(item, data, drop_none, drop_ellipsis)
        elif isinstance(item, FunctionData):
            process_func_def(item, data)

    return ''.join(data.lines)


def skip_blank(d: FileData, line: int) -> int:
    """Return first non-blank line number after `line`."""
    while not d.lines[line].strip():
        line += 1
    return line


def find_start(d: FileData, line_com: int) -> int:
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


def check_target(assign: ast.Assign) -> bool:
    """Check if the statement is suitable for annotation.

    Type comments can placed on with and for statements, but
    annotation can be placed only on an simple assignment with a single target.
    """
    if len(assign.targets) == 1:
        target = assign.targets[0]
    else:
        return False
    if (
        isinstance(target, ast.Name) or isinstance(target, ast.Attribute) or
        isinstance(target, ast.Subscript)
    ):
        return True
    return False


def find_eq(d: FileData, line_start: int) -> Tuple[int, int]:
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


def find_val(d: FileData, pos_eq: Tuple[int, int]) -> Tuple[int, int]:
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


def find_target(d: FileData, pos_eq: Tuple[int, int]) -> Tuple[int, int]:
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
        # We want to work only with file without syntax errors
        tree = ast.parse(code, type_comments=True)
    except SyntaxError:
        return None
    lines = code.splitlines(keepends=True)
    rl = BytesIO(code.encode('utf-8')).readline
    tokens = list(tokenize.tokenize(rl))

    data = FileData(lines, tokens, tree)
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
