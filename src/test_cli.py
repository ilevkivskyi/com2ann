from __future__ import annotations

import os
import pathlib
from typing import Any, Callable, Iterable, TypedDict

import pytest

import com2ann


@pytest.fixture
def test_path(tmp_path: pathlib.Path) -> Iterable[pathlib.Path]:
    old_path = pathlib.Path.cwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_path)


class Exited(Exception):
    pass


class ParseResult(TypedDict, total=False):
    status: int
    args: dict[str, Any]
    error: bool
    out: str
    err: str


ParseCallable = Callable[..., ParseResult]


@pytest.fixture
def parse(
    capsys: pytest.CaptureFixture[str],
) -> ParseCallable:
    # parse is a pytext fixture returning a function
    # This function will:
    # - Ensure that SystemExit exceptions raised by ArgParse are
    #   caught before they exit pytest
    # - Save the result of the arg parsing if successful
    # - Save the status code otherwise
    # - Save and print whatever argparse printed to stdout and stderr

    def _(*args: str) -> ParseResult:
        result: ParseResult = {"status": 0}
        try:
            result.update({"args": com2ann.parse_cli_args(args), "error": False})
        except SystemExit as exc:
            result.update({"status": exc.code})

        out, err = capsys.readouterr()
        # Will only display if the test fails
        if out:
            print("stdout:", out)
            result["out"] = out
        if err:
            print("stderr:", err)
            result["err"] = err

        return result

    return _


def test_parse_cli_args__minimal(parse: ParseCallable, test_path: pathlib.Path) -> None:
    (test_path / "a.py").touch()

    assert parse("a.py")["args"] == {
        "add_future_imports": False,
        "drop_ellipsis": False,
        "drop_none": False,
        "infile": [pathlib.Path("a.py")],
        "outfile": None,
        "python_minor_version": -1,
        "silent": False,
        "wrap_signatures": 0,
    }


def test_parse_cli_args__maximal(parse: ParseCallable, test_path: pathlib.Path) -> None:
    (test_path / "a.py").touch()

    assert parse(
        "a.py",
        "--outfile=b.py",
        "--add-future-imports",
        "--drop-ellipsis",
        "--drop-none",
        "--python-minor-version=7",
        "--silent",
        "--wrap-signatures=88",
    )["args"] == {
        "add_future_imports": True,
        "drop_ellipsis": True,
        "drop_none": True,
        "infile": [pathlib.Path("a.py")],
        "outfile": pathlib.Path("b.py"),
        "python_minor_version": 7,
        "silent": True,
        "wrap_signatures": 88,
    }


def test_parse_cli_args__no_infile(parse: ParseCallable) -> None:
    result = parse()

    assert result == {"err": "No input file, exiting", "status": 0}


def test_parse_cli_args__missing_file(parse: ParseCallable) -> None:

    result = parse("a.py")
    assert result["status"] == 2
    assert "File(s) not found: a.py" in result["err"]


@pytest.mark.parametrize(
    "infile, outfile",
    [
        ("file.py", "dir"),
        ("dir", "file.py"),
    ],
)
def test_parse_cli_args__in_out_type_mismatch(
    parse: ParseCallable, test_path: pathlib.Path, infile: str, outfile: str
) -> None:
    (test_path / "file.py").touch()
    (test_path / "dir").mkdir()

    result = parse(infile, "--outfile", outfile)
    assert result["status"] == 2
    assert "Infile must be the same type" in result["err"]


def test_parse_cli_args__outfile_doesnt_exist(
    parse: ParseCallable, test_path: pathlib.Path
) -> None:
    (test_path / "dir").mkdir()

    result = parse("dir", "--outfile", "other_dir")
    assert result["status"] == 0


def test_parse_cli_args__multiple_inputs_and_output(
    parse: ParseCallable, test_path: pathlib.Path
) -> None:
    (test_path / "a.py").touch()
    (test_path / "b.py").touch()

    result = parse("a.py", "b.py", "--outfile", "c.py")
    assert result["status"] == 2
    assert "Cannot use --outfile if multiple infiles are given" in result["err"]


def test_rebase_path() -> None:
    assert com2ann.rebase_path(
        path=pathlib.Path("a/b/c/d"),
        root=pathlib.Path("a/b/"),
        new_root=pathlib.Path("e/f"),
    ) == pathlib.Path("e/f/c/d")


@pytest.fixture
def options() -> com2ann.Options:
    return com2ann.Options(drop_none=True, drop_ellipsis=True, silent=False)


@pytest.fixture
def translate_file(mocker: Any) -> Any:
    return mocker.patch("com2ann.translate_file", autospec=True)


def test_process_single_entry__file(
    test_path: pathlib.Path, translate_file: Any, options: com2ann.Options
) -> None:
    in_path = test_path / "a.py"
    in_path.touch()

    out_path = test_path / "b.py"

    com2ann.process_single_entry(in_path=in_path, out_path=out_path, options=options)

    translate_file.assert_called_with(
        infile=str(in_path),
        outfile=str(out_path),
        options=options,
    )


def test_process_single_entry__dir(
    test_path: pathlib.Path, translate_file: Any, options: com2ann.Options, mocker: Any
) -> None:
    in_path = test_path / "a"
    (in_path / "c/d/e").mkdir(parents=True)
    (in_path / "c/d/e/f.txt").touch()
    (in_path / "c/d/e/f.py").touch()
    (in_path / "c/d/e/f.pyi").touch()
    (in_path / "c/d/e/f.pyx").touch()

    out_path = test_path / "b"

    com2ann.process_single_entry(in_path=in_path, out_path=out_path, options=options)

    assert translate_file.mock_calls == [
        mocker.call(
            infile=str(in_path / "c/d/e/f.py"),
            outfile=str(out_path / "c/d/e/f.py"),
            options=options,
        ),
        mocker.call(
            infile=str(in_path / "c/d/e/f.pyi"),
            outfile=str(out_path / "c/d/e/f.pyi"),
            options=options,
        ),
    ]
