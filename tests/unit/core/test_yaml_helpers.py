from collections import OrderedDict

import pytest
import yaml

from ppdet_pytorch.core.config.yaml_helpers import Callable, setup_orderdict


def test_callable_defaults_are_isolated_between_config_objects():
    first = Callable("len")
    second = Callable("len")

    first.args.append([1, 2])
    first.kwargs["unexpected"] = True

    assert second.args == []
    assert second.kwargs == {}


def test_callable_invokes_builtins_and_qualified_functions():
    assert Callable("len", args=[[1, 2, 3]])() == 3
    assert Callable("operator.add", args=[2, 5])() == 7


def test_callable_yaml_round_trip_supports_mapping_and_sequence_nodes():
    mapping_value = yaml.load(
        "!Callable\nfull_type: builtins.sorted\nargs: [[3, 1, 2]]\n"
        "kwargs: {reverse: true}\n",
        Loader=yaml.Loader,
    )
    sequence_value = yaml.load("!Callable [len, [[a, b]]]", Loader=yaml.Loader)
    round_tripped = yaml.load(yaml.dump(mapping_value), Loader=yaml.Loader)

    assert mapping_value() == [3, 2, 1]
    assert sequence_value() == 2
    assert round_tripped() == [3, 2, 1]


def test_callable_yaml_reports_invalid_constructor_arguments(capsys):
    with pytest.raises(TypeError):
        yaml.load("!Callable {full_type: len, unknown: true}", Loader=yaml.Loader)

    assert "Error when construct Callable instance" in capsys.readouterr().out


def test_ordered_dict_representer_preserves_declared_order():
    setup_orderdict()

    dumped = yaml.dump(OrderedDict([("second", 2), ("first", 1)]), sort_keys=False)

    assert dumped.index("second:") < dumped.index("first:")
