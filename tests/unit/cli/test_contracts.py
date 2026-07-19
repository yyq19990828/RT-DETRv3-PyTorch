import pytest

from ppdet_pytorch.cli import convert as convert_cli
from ppdet_pytorch.cli import eval as eval_cli
from ppdet_pytorch.cli import export as export_cli
from ppdet_pytorch.cli import infer as infer_cli
from ppdet_pytorch.cli import models as models_cli
from ppdet_pytorch.cli import train as train_cli


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
