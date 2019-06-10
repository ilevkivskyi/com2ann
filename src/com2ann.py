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
import sys
import argparse
import tokenize
from tokenize import TokenInfo
from collections import defaultdict
from io import BytesIO
from dataclasses import dataclass

from typing import List, DefaultDict, Tuple, Optional, Union

__all__ = ['com2ann', 'TYPE_COM']

TYPE_COM = re.compile(r'\s*#\s*type\s*:(.*)$', flags=re.DOTALL)

# For internal use only.
_TRAILER = re.compile(r'\s*$', flags=re.DOTALL)
_NICE_IGNORE = re.compile(r'\s*# type: ignore\s*$', flags=re.DOTALL)

Unsupported = Union[ast.For, ast.AsyncFor, ast.With, ast.AsyncWith]
Function = Union[ast.FunctionDef, ast.AsyncFunctionDef]


@dataclass
class Options:
    drop_none: bool
    drop_ellipsis: bool
    silent: bool
    wrap_signatures: int = 0
    python_minor_version: int = -1


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
    def __init__(self, silent: bool) -> None:
        super().__init__()
        self.silent = silent
        self.found: List[Union[AssignData, FunctionData]] = []
        self.found_unsupported: List[int] = []

    def visit_Assign(self, s: ast.Assign) -> None:
        if s.type_comment:
            if not check_target(s):
                self.found_unsupported.append(s.lineno)
                return
            target = s.targets[0]
            value = s.value

            tuple_rvalue = isinstance(value, ast.Tuple)
            none_rvalue = isinstance(value, ast.Constant) and value.value is None
            ellipsis_rvalue = isinstance(value, ast.Constant) and value.value is Ellipsis

            assert (target.end_lineno and target.end_col_offset and
                    value.end_lineno and value.end_col_offset)
            found = AssignData(s.type_comment,
                               target.end_lineno, target.end_col_offset,
                               value.lineno, value.col_offset,
                               value.end_lineno, value.end_col_offset,
                               tuple_rvalue, none_rvalue, ellipsis_rvalue)
            self.found.append(found)

    def visit_For(self, o: ast.For) -> None:
        self.visit_unsupported(o)

    def visit_AsyncFor(self, o: ast.AsyncFor) -> None:
        self.visit_unsupported(o)

    def visit_With(self, o: ast.With) -> None:
        self.visit_unsupported(o)

    def visit_AsyncWith(self, o: ast.AsyncWith) -> None:
        self.visit_unsupported(o)

    def visit_unsupported(self, o: Unsupported) -> None:
        if o.type_comment:
            self.found_unsupported.append(o.lineno)

    def visit_FunctionDef(self, fdef: ast.FunctionDef) -> None:
        self.visit_function_impl(fdef)

    def visit_AsyncFunctionDef(self, fdef: ast.AsyncFunctionDef) -> None:
        self.visit_function_impl(fdef)

    def visit_function_impl(self, fdef: Function) -> None:
        if (fdef.type_comment or
                any(a.type_comment for a in fdef.args.args) or
                any(a.type_comment for a in fdef.args.kwonlyargs) or
                fdef.args.vararg and fdef.args.vararg.type_comment or
                fdef.args.kwarg and fdef.args.kwarg.type_comment):
            num_non_defs = len(fdef.args.args) - len(fdef.args.defaults)
            num_kw_non_defs = (len(fdef.args.kwonlyargs) -
                               len([d for d in fdef.args.kw_defaults if d is not None]))

            args = self.process_per_arg_comments(fdef, num_non_defs, num_kw_non_defs)

            ret: Optional[str]
            if fdef.type_comment:
                res = split_function_comment(fdef.type_comment, self.silent)
                if not res:
                    self.found_unsupported.append(fdef.lineno)
                    return
                f_args, ret = res
            else:
                f_args, ret = [], None

            if args and f_args:
                if not self.silent:
                    print(f'Both per-argument and function comments for "{fdef.name}"',
                          file=sys.stderr)
                self.found_unsupported.append(fdef.lineno)
                return

            body_start = fdef.body[0].lineno
            if isinstance(fdef.body[0], (ast.AsyncFunctionDef,
                                         ast.FunctionDef,
                                         ast.ClassDef)):
                if fdef.body[0].decorator_list:
                    body_start = min(it.lineno for it in fdef.body[0].decorator_list)
            if args:
                self.found.append(FunctionData(args, ret, fdef.lineno, body_start))
            elif not f_args:
                self.found.append(FunctionData([], ret, fdef.lineno, body_start))
            else:
                c_args = self.process_function_comment(fdef, f_args,
                                                       num_non_defs, num_kw_non_defs)
                if c_args is None:
                    # There was an error.
                    return
                self.found.append(FunctionData(c_args, ret, fdef.lineno, body_start))
        self.generic_visit(fdef)

    def process_per_arg_comments(self, fdef: Function,
                                 num_non_defs: int,
                                 num_kw_non_defs: int) -> List[ArgComment]:
        args: List[ArgComment] = []

        for i, a in enumerate(fdef.args.args):
            if a.type_comment:
                assert a.end_col_offset
                args.append(ArgComment(a.type_comment,
                                       a.lineno, a.end_col_offset,
                                       i >= num_non_defs))
        if fdef.args.vararg and fdef.args.vararg.type_comment:
            vararg = fdef.args.vararg
            assert vararg.end_col_offset
            args.append(ArgComment(fdef.args.vararg.type_comment,
                                   vararg.lineno, vararg.end_col_offset,
                                   False))

        for i, a in enumerate(fdef.args.kwonlyargs):
            if a.type_comment:
                assert a.end_col_offset
                args.append(ArgComment(a.type_comment,
                                       a.lineno, a.end_col_offset,
                                       i >= num_kw_non_defs))
        if fdef.args.kwarg and fdef.args.kwarg.type_comment:
            kwarg = fdef.args.kwarg
            assert kwarg.end_col_offset
            args.append(ArgComment(fdef.args.kwarg.type_comment,
                                   kwarg.lineno, kwarg.end_col_offset,
                                   False))
        return args

    def process_function_comment(self, fdef: Function,
                                 f_args: List[str],
                                 num_non_defs: int,
                                 num_kw_non_defs: int) -> Optional[List[ArgComment]]:
        args: List[ArgComment] = []

        tot_args = len(fdef.args.args) + len(fdef.args.kwonlyargs)
        if fdef.args.vararg:
            tot_args += 1
        if fdef.args.kwarg:
            tot_args += 1

        if len(f_args) not in (tot_args, tot_args - 1):
            if not self.silent:
                print(f'Invalid number of arguments in comment for "{fdef.name}"',
                      file=sys.stderr)
            self.found_unsupported.append(fdef.lineno)
            return None

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

            kwonlyargs = fdef.args.kwonlyargs
            if a in kwonlyargs and kwonlyargs.index(a) >= num_kw_non_defs:
                has_default = True

            assert a.end_col_offset
            args.append(ArgComment(typ,
                                   a.lineno, a.end_col_offset,
                                   has_default))
        return args


def split_sub_comment(comment: str) -> Tuple[str, Optional[str]]:
    """Split extra comment from a type comment.

    The only non-trivial thing here is to take care of literal types,
    that can contain arbitrary chars, including '#'.
    """
    rl = BytesIO(comment.encode('utf-8')).readline
    tokens = list(tokenize.tokenize(rl))

    i_sub = None
    for i, tok in enumerate(tokens):
        if tok.exact_type == tokenize.COMMENT:
            _, i_sub = tokens[i - 1].end

    if i_sub is not None:
        return comment[:i_sub], comment[i_sub:]
    return comment, None


def split_function_comment(comment: str,
                           silent: bool = False) -> Optional[Tuple[List[str], str]]:
    typ, _ = split_sub_comment(comment)
    if '->' not in typ:
        if not silent:
            print('Invalid function type comment:', comment,
                  file=sys.stderr)
        return None

    # TODO: ()->int vs () -> int -- keep spacing (also # type:int vs # type: int).
    arg_list, ret = typ.split('->')

    arg_list = arg_list.strip()
    ret = ret.strip()

    if not(arg_list[0] == '(' and arg_list[-1] == ')'):
        if not silent:
            print('Invalid function type comment:', comment,
                  file=sys.stderr)
        return None

    arg_list = arg_list[1:-1]

    args: List[str] = []

    # TODO: use tokenizer to guard against Literal[','].
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
    assert match, line
    if match.group(1).lstrip().startswith('ignore'):
        # Keep # type: ignore[=code] comments.
        return line
    rest = line[:match.start()]

    typ = match.group(1)
    _, sub_comment = split_sub_comment(typ)
    if sub_comment is None:
        trailer = re.search(_TRAILER, typ)
        assert trailer
        sub_comment = typ[trailer.start():]

    if rest:
        new_line = rest + sub_comment
    else:
        # A type comment on line of its own.
        new_line = line[:line.index('#')] + sub_comment.lstrip(' \t')
    return new_line


def string_insert(line: str, extra: str, pos: int) -> str:
    return line[:pos] + extra + line[pos:]


def process_assign(comment: AssignData, data: FileData,
                   drop_none: bool, drop_ellipsis: bool) -> None:
    lines = data.lines

    rv_end = comment.rvalue_end_line - 1
    rv_start = comment.rvalue_start_line - 1
    lines[rv_end] = strip_type_comment(lines[rv_end])

    if comment.tuple_rvalue:
        # TODO: take care of (1, 2), (3, 4) with matching pars.
        if not (lines[rv_start][comment.rvalue_start_offset] == '(' and
                lines[rv_end][comment.rvalue_end_offset - 1] == ')'):
            # We need to wrap rvalue in parentheses before Python 3.8.
            end_line = lines[rv_end]
            lines[rv_end] = string_insert(end_line, ')',
                                          comment.rvalue_end_offset)

            start_line = lines[rv_start]
            lines[rv_start] = string_insert(start_line, '(',
                                            comment.rvalue_start_offset)

            if comment.rvalue_end_line > comment.rvalue_start_line:
                # Add a space to fix indentation after inserting paren.
                for i in range(comment.rvalue_end_line, comment.rvalue_start_line, -1):
                    if lines[i - 1].strip():
                        lines[i - 1] = ' ' + lines[i - 1]

    elif comment.none_rvalue and drop_none or comment.ellipsis_rvalue and drop_ellipsis:
        # TODO: more tricky (multi-line) cases.
        assert comment.lvalue_end_line == comment.rvalue_end_line
        line = lines[comment.lvalue_end_line - 1]
        lines[comment.lvalue_end_line - 1] = (line[:comment.lvalue_end_offset] +
                                              line[comment.rvalue_end_offset:])

    lvalue_line = lines[comment.lvalue_end_line - 1]

    typ, _ = split_sub_comment(comment.type_comment)

    # TODO: this is pretty ad hoc.
    lv_line = lines[comment.lvalue_end_line - 1]
    while (comment.lvalue_end_offset < len(lv_line) and
           lv_line[comment.lvalue_end_offset] == ')'):
        comment.lvalue_end_offset += 1

    lines[comment.lvalue_end_line - 1] = (lvalue_line[:comment.lvalue_end_offset] +
                                          ': ' + typ +
                                          lvalue_line[comment.lvalue_end_offset:])


def insert_arg_type(line: str, arg: ArgComment) -> str:
    typ, _ = split_sub_comment(arg.type_comment)

    new_line = line[:arg.arg_end_offset] + ': ' + typ

    rest = line[arg.arg_end_offset:]
    if not arg.has_default:
        return new_line + rest

    # Here we are a bit opinionated about spacing (see PEP 8).
    rest = rest.lstrip()
    assert rest[0] == '='
    rest = rest[1:].lstrip()

    return new_line + ' = ' + rest


def wrap_function_header(header: str) -> List[str]:
    # TODO: use tokenizer to guard against Literal[','].
    parts: List[str] = []
    next_part = ''
    nested = 0
    indent: Optional[int] = None

    for i, c in enumerate(header):
        if c in '([{':
            nested += 1
            if c == '(' and indent is None:
                indent = i + 1
        if c in ')]}':
            nested -= 1
        if c == ',' and nested == 1:
            next_part += c
            parts.append(next_part)
            next_part = ''
        else:
            next_part += c

    parts.append(next_part)

    if len(parts) == 1:
        return parts

    # Indent all the wrapped lines.
    assert indent is not None
    parts = [parts[0]] + [' ' * indent + p.lstrip(' \t') for p in parts[1:]]

    # Add end lines like in the original header.
    trailer = re.search(_TRAILER, header)
    assert trailer
    end_line = header[trailer.start():].lstrip(' \t')
    parts = [p + end_line for p in parts[:-1]] + [parts[-1]]

    # TODO: handle type ignores better.
    ignore = re.search(_NICE_IGNORE, parts[-1])
    if ignore:
        # We should keep # type: ignore on the first line of the wrapped header.
        last = parts[-1]
        first = parts[0]
        first_trailer = re.search(_TRAILER, first)
        assert first_trailer
        parts[0] = first[:first_trailer.start()] + ignore.group()
        parts[-1] = last[:ignore.start()] + first_trailer.group()

    return parts


def process_func_def(func_type: FunctionData, data: FileData, wrap_sig: int) -> None:
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
        lines[ret_line] = (lines[ret_line][:right_par + 1] +
                           ' -> ' + func_type.ret_type +
                           lines[ret_line][colon:])

    for arg in reversed(func_type.arg_types):
        lines[arg.arg_line - 1] = insert_arg_type(lines[arg.arg_line - 1], arg)

    if ret_line == func_type.header_start_line - 1:
        header = data.lines[ret_line]
        if wrap_sig and len(header) > wrap_sig:
            data.lines[ret_line:ret_line + 1] = wrap_function_header(header)


def com2ann_impl(data: FileData, drop_none: bool, drop_ellipsis: bool,
                 wrap_sig: int = 0, silent: bool = True) -> str:
    finder = TypeCommentCollector(silent)
    finder.visit(data.tree)

    data.fail.extend(finder.found_unsupported)
    found = list(reversed(finder.found))

    for item in found:
        if isinstance(item, AssignData):
            process_assign(item, data, drop_none, drop_ellipsis)
            data.success.append(item.lvalue_end_line)
        elif isinstance(item, FunctionData):
            process_func_def(item, data, wrap_sig)
            data.success.append(item.header_start_line)

    return ''.join(data.lines)


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


def com2ann(code: str, *,
            drop_none: bool = False,
            drop_ellipsis: bool = False,
            silent: bool = False,
            wrap_sig: int = 0,
            python_minor_version: int = -1) -> Optional[Tuple[str, FileData]]:
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
        tree = ast.parse(code,
                         type_comments=True,
                         feature_version=python_minor_version)
    except SyntaxError:
        return None
    lines = code.splitlines(keepends=True)
    rl = BytesIO(code.encode('utf-8')).readline
    tokens = list(tokenize.tokenize(rl))

    data = FileData(lines, tokens, tree)
    new_code = com2ann_impl(data, drop_none, drop_ellipsis, wrap_sig, silent)

    if not silent:
        if data.success:
            print('Comments translated for statements on lines:',
                  ', '.join(str(lno) for lno in data.success))
        if data.fail:
            print('Comments skipped for statements on lines:',
                  ', '.join(str(lno) for lno in data.fail))
        if not data.success and not data.fail:
            print('No type comments found')

    return new_code, data


def translate_file(infile: str, outfile: str, options: Options) -> None:
    try:
        opened = tokenize.open(infile)
    except SyntaxError:
        print("Cannot open", infile, file=sys.stderr)
        return
    with opened as f:
        code = f.read()
        enc = f.encoding
    if not options.silent:
        print('File:', infile)
    result = com2ann(code,
                     drop_none=options.drop_none,
                     drop_ellipsis=options.drop_ellipsis,
                     silent=options.silent,
                     wrap_sig=options.wrap_signatures,
                     python_minor_version=options.python_minor_version)
    if result is None:
        print("SyntaxError in", infile, file=sys.stderr)
        return
    new_code, _ = result
    with open(outfile, 'wb') as fo:
        fo.write(new_code.encode(enc))


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
    parser.add_argument("-w", "--wrap-signatures",
                        help="Wrap function headers that are longer than given length",
                        type=int, default=0)
    parser.add_argument("-v", "--python-minor-version",
                        help="Python 3 minor version to use to parse the files",
                        type=int, default=-1)

    args = parser.parse_args()
    if args.outfile is None:
        args.outfile = args.infile

    options = Options(args.drop_none, args.drop_ellipsis,
                      args.silent, args.wrap_signatures,
                      args.python_minor_version)

    if os.path.isfile(args.infile):
        translate_file(args.infile, args.outfile, options)
    else:
        for root, _, files in os.walk(args.infile):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == '.py' or ext == '.pyi':
                    file_name = os.path.join(root, file)
                    translate_file(file_name, file_name, options)
