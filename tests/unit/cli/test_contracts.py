import pytest

from detrs.cli import convert as convert_cli
from detrs.cli import eval as eval_cli
from detrs.cli import export as export_cli
from detrs.cli import infer as infer_cli
from detrs.cli import main as main_cli
from detrs.cli import models as models_cli
from detrs.cli import train as train_cli


@pytest.mark.parametrize(
    "parser_factory",
    [
        train_cli.create_argument_parser,
        eval_cli.create_argument_parser,
        infer_cli.create_argument_parser,
        convert_cli.create_argument_parser,
        export_cli.create_argument_parser,
        models_cli.create_argument_parser,
    ],
)
def test_public_cli_help_exits_successfully(parser_factory, capsys):
    with pytest.raises(SystemExit) as error:
        parser_factory().parse_args(["--help"])

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", list(main_cli.COMMANDS))
def test_dispatch_parser_registers_command_help(command, capsys):
    parser = main_cli.create_argument_parser(command)

    with pytest.raises(SystemExit) as error:
        parser.parse_args([command, "--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert f"usage: detrs {command}" in output


def test_dispatch_help_lists_all_commands(capsys):
    assert main_cli.main(["--help"]) == 0

    output = capsys.readouterr().out
    for command in main_cli.COMMANDS:
        assert command in output


def test_dispatch_rejects_unknown_command(capsys):
    with pytest.raises(SystemExit) as error:
        main_cli.main(["not-a-command"])

    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
