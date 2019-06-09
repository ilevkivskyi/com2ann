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

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

__all__ = ['com2ann', 'TYPE_COM']

TYPE_COM = re.compile(r'\s*#\s*type\s*:.*$', flags=re.DOTALL)
TRAIL_OR_COM = re.compile(r'\s*$|\s*#.*$', flags=re.DOTALL)


# TODO: use tokenizer for split_function_comment() and split_sub_comment().

def split_function_comment(comment: str) -> Tuple[List[str], str]:
    # TODO: ()->int vs () -> int -- preserve spacing (maybe also # type:int vs # type: int)
    # TODO: fail gracefully on invalid types.
    typ = comment.split('#', maxsplit=1)[0].rstrip()
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


@dataclass
class Data:
    success: List[int]
    fail: List[int]


@dataclass
class AssignComment:
    type_comment: str
    lvalue_end_line: int
    lvalue_end_offset: int
    rvalue_end_line: int


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


class TypeCommentCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.found: List[Union[AssignComment, FunctionData]] = []

    def visit_Assign(self, s: ast.Assign) -> None:
        if s.type_comment:
            # TODO: what if more targets?
            target = s.targets[0]
            found = AssignComment(s.type_comment,
                                  target.end_lineno, target.end_col_offset,
                                  s.value.end_lineno)
            self.found.append(found)

    def visit_FunctionDef(self, fdef: ast.FunctionDef) -> None:
        if (fdef.type_comment or
                any(a.type_comment for a in fdef.args.args) or
                any(a.type_comment for a in fdef.args.kwonlyargs) or
                fdef.args.vararg and fdef.vararg.type_comment or
                fdef.args.kwarg and fdef.args.kwarg.type_comment):
            num_non_defs = len(fdef.args.args) - len(fdef.args.defaults)
            num_kw_non_defs = len(fdef.args.kwonlyargs) - len([d for d in fdef.args.kw_defaults if d is not None])

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
                    iter_args = fdef.args.args

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

                self.found.append(FunctionData(args, ret, fdef.lineno, fdef.body[0].lineno))
        self.generic_visit(fdef)


def strip_type_comment(line: str) -> str:
    match = re.search(TYPE_COM, line)
    assert match
    matched = line[match.start():]
    matched = matched.lstrip()[1:]

    rest = line[:match.start()]
    sub_comment = re.search(TRAIL_OR_COM, matched)
    assert sub_comment
    if rest:
        new_line = rest + matched[sub_comment.start():]
    else:
        # A type comment on line of its own.
        new_line = line[:line.index('#')] + matched[sub_comment.start():].lstrip(' \t')
    return new_line


def process_assign(lines: List[str], comment: AssignComment, data: Data) -> None:
    lines[comment.rvalue_end_line - 1] = strip_type_comment(lines[comment.rvalue_end_line - 1])

    lvalue_line = lines[comment.lvalue_end_line - 1]
    # TODO: take care of Literal['#'].
    typ = comment.type_comment.split('#', maxsplit=1)[0].rstrip()
    lines[comment.lvalue_end_line - 1] = (lvalue_line[:comment.lvalue_end_offset] +
                                          ': ' + typ +
                                          lvalue_line[comment.lvalue_end_offset:])


def insert_arg_type(line: str, arg: ArgComment) -> str:
    typ = arg.type_comment.split('#', maxsplit=1)[0].rstrip()

    new_line = line[:arg.arg_end_offset] + ': ' + typ

    rest = line[arg.arg_end_offset:]
    if not arg.has_default:
        return new_line + rest

    # Here we are a bit opinionated about spacing (see PEP 8).
    rest = rest.lstrip()
    assert rest[0] == '=', (line, rest)
    rest = rest[1:].lstrip()

    return new_line + ' = ' + rest


def process_func_def(lines: List[str], func_type: FunctionData, data: Data) -> None:
    removed = 0
    for i in range(func_type.body_first_line - 2, func_type.header_start_line - 2, -1):
        if re.search(TYPE_COM, lines[i]):
            lines[i] = strip_type_comment(lines[i])
            if not lines[i].strip():
                removed += 1
                del lines[i]

    # Inserting return type is a bit dirty...
    if func_type.ret_type:
        ret_line = func_type.body_first_line - removed - 1
        ret_line -= 1
        while not lines[ret_line].split('#')[0].strip():
            ret_line -= 1

        # TODO: use also tokenizer here to take care of possible comment.
        colon = lines[ret_line].rindex(':')
        right_par = lines[ret_line][:colon].rindex(')')
        lines[ret_line] = lines[ret_line][:right_par + 1] + ' -> ' + func_type.ret_type + lines[ret_line][colon:]

    for arg in reversed(func_type.arg_types):
        lines[arg.arg_line - 1] = insert_arg_type(lines[arg.arg_line - 1], arg)


def com2ann_impl(code: str, drop_none: bool, drop_ellipsis: bool) -> Optional[Tuple[str, Data]]:
    lines = code.splitlines(keepends=True)

    try:
        tree = ast.parse(code, type_comments=True)
    except SyntaxError:
        return None

    finder = TypeCommentCollector()
    finder.visit(tree)

    found = list(reversed(finder.found))

    data = Data([], [])
    for item in found:
        if isinstance(item, AssignComment):
            process_assign(lines, item, data)
        elif isinstance(item, FunctionData):
            process_func_def(lines, item, data)

    return ''.join(lines), data


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

    new_code, data = com2ann_impl(code, drop_none, drop_ellipsis)

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
