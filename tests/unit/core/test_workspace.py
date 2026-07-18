import pytest

from ppdet_pytorch.core.workspace import (
    create,
    global_config,
    merge_config,
    register,
)


@pytest.fixture
def registered_components():
    names = []

    def register_for_test(cls):
        register(cls)
        names.append(cls.__name__)
        return cls

    yield register_for_test

    for name in names:
        global_config.pop(name, None)
    global_config.pop("WorkspaceNamedBlock", None)


def test_create_accepts_named_config_and_explicit_kwargs(registered_components):
    @registered_components
    class WorkspaceValue:
        def __init__(self, value=1):
            self.value = value

    instance = create({"name": "WorkspaceValue", "value": 2}, value=3)

    assert isinstance(instance, WorkspaceValue)
    assert instance.value == 3


def test_create_resolves_named_global_config_block(registered_components):
    @registered_components
    class WorkspaceDataset:
        def __init__(self, root, sample_num=-1):
            self.root = root
            self.sample_num = sample_num

    global_config["WorkspaceNamedBlock"] = {
        "name": "WorkspaceDataset",
        "root": "fixture",
        "sample_num": 2,
    }

    instance = create("WorkspaceNamedBlock")

    assert isinstance(instance, WorkspaceDataset)
    assert instance.root == "fixture"
    assert instance.sample_num == 2


def test_create_uses_context_for_from_config_without_leaking_it(
    registered_components,
):
    @registered_components
    class WorkspaceFromConfig:
        def __init__(self, value=1, channels=None):
            self.value = value
            self.channels = channels

        @classmethod
        def from_config(cls, cfg, input_shape):
            return {"channels": list(input_shape)}

    instance = create(
        {"name": "WorkspaceFromConfig", "value": 2},
        input_shape=(16, 32),
        value=4,
    )

    assert instance.value == 4
    assert instance.channels == [16, 32]


def test_create_rejects_config_without_component_name():
    with pytest.raises(ValueError, match="name.*type"):
        create({"value": 1})


def test_merge_config_respects_empty_explicit_target(isolated_workspace):
    key = "__workspace_merge_target__"
    target = {}

    result = merge_config({key: {"value": 1}}, target)

    assert result is target
    assert target[key] == {"value": 1}
    assert key not in global_config
