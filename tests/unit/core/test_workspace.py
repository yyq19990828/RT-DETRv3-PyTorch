import pytest
import yaml

from ppdet_pytorch.core.config.schema import extract_schema
from ppdet_pytorch.core.workspace import (
    create,
    global_config,
    load_config,
    merge_config,
    register,
)


def test_schema_type_validation_supports_typeguard_four_signature():
    class WorkspaceTyped:
        def __init__(self, width: int, label: str = "default"):
            self.width = width
            self.label = label

    schema = extract_schema(WorkspaceTyped)

    assert schema.name == "WorkspaceTyped"
    assert schema.cls is WorkspaceTyped
    assert schema["label"] == "default"

    schema["width"] = 3
    assert schema.find_mismatch_keys() == []
    schema.validate()

    schema["width"] = "3"
    assert schema.find_mismatch_keys() == ["width"]
    with pytest.raises(TypeError, match="Wrong param type.*width"):
        schema.validate()


@pytest.fixture
def registered_components(isolated_workspace):
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


def test_shared_resolution_priority_is_explicit_global_default(
    registered_components,
):
    @registered_components
    class WorkspaceShared:
        __shared__ = ["width"]

        def __init__(self, width=10):
            self.width = width

    assert create("WorkspaceShared").width == 10

    global_config["width"] = 20
    assert create("WorkspaceShared").width == 20
    assert create({"name": "WorkspaceShared", "width": 30}).width == 30
    assert (
        create(
            {"name": "WorkspaceShared", "width": 30},
            width=40,
        ).width
        == 40
    )


def test_explicit_constructor_value_wins_over_from_config(
    registered_components,
):
    @registered_components
    class WorkspaceFromConfigConflict:
        def __init__(self, value="default", context=None):
            self.value = value
            self.context = context

        @classmethod
        def from_config(cls, cfg, input_shape):
            return {
                "value": "from_config",
                "context": tuple(input_shape),
            }

    from_config_instance = create(
        {"name": "WorkspaceFromConfigConflict", "value": "component"},
        input_shape=(8, 16),
    )
    explicit_instance = create(
        {"name": "WorkspaceFromConfigConflict", "value": "component"},
        input_shape=(8, 16),
        value="explicit",
    )

    assert from_config_instance.value == "from_config"
    assert from_config_instance.context == (8, 16)
    assert explicit_instance.value == "explicit"


def test_explicit_injection_config_is_resolved_before_construction(
    registered_components,
):
    @registered_components
    class WorkspaceDependency:
        def __init__(self, value=1):
            self.value = value

    @registered_components
    class WorkspaceConsumer:
        __inject__ = ["dependency"]

        def __init__(self, dependency=None):
            self.dependency = dependency

    instance = create(
        {
            "name": "WorkspaceConsumer",
            "dependency": {"name": "WorkspaceDependency", "value": 2},
        },
        dependency={"name": "WorkspaceDependency", "value": 3},
    )

    assert isinstance(instance.dependency, WorkspaceDependency)
    assert instance.dependency.value == 3


def test_component_injection_wins_over_from_config(registered_components):
    @registered_components
    class WorkspaceConfiguredDependency:
        def __init__(self, value):
            self.value = value

    @registered_components
    class WorkspaceConfiguredConsumer:
        __inject__ = ["dependency"]

        def __init__(self, dependency=None):
            self.dependency = dependency

        @classmethod
        def from_config(cls, cfg):
            return {"dependency": WorkspaceConfiguredDependency("from_config")}

    instance = create(
        {
            "name": "WorkspaceConfiguredConsumer",
            "dependency": {
                "name": "WorkspaceConfiguredDependency",
                "value": "component",
            },
        }
    )

    assert isinstance(instance.dependency, WorkspaceConfiguredDependency)
    assert instance.dependency.value == "component"


def test_repeated_load_config_replaces_previous_runtime_values(
    registered_components,
    tmp_path,
):
    @registered_components
    class WorkspaceReloaded:
        def __init__(self, alpha=0, beta=0):
            self.alpha = alpha
            self.beta = beta

    first_path = tmp_path / "first.yml"
    first_path.write_text(
        "first_only: true\n"
        "WorkspaceAlias:\n"
        "  name: WorkspaceReloaded\n"
        "  alpha: 7\n"
        "WorkspaceReloaded:\n"
        "  alpha: 1\n"
        "  beta: 2\n",
        encoding="utf-8",
    )
    second_path = tmp_path / "second.yml"
    second_path.write_text(
        "WorkspaceReloaded:\n  alpha: 3\n",
        encoding="utf-8",
    )

    first_snapshot = load_config(first_path).copy()
    second_config = load_config(second_path)
    instance = create("WorkspaceReloaded")

    assert first_snapshot["first_only"] is True
    assert first_snapshot["WorkspaceReloaded"]["beta"] == 2
    assert "first_only" not in second_config
    assert "WorkspaceAlias" not in second_config
    assert second_config.filename == "second"
    assert instance.alpha == 3
    assert instance.beta == 0


def test_failed_load_config_preserves_active_workspace(
    registered_components,
    tmp_path,
):
    @registered_components
    class WorkspacePreserved:
        def __init__(self, value=0):
            self.value = value

    valid_path = tmp_path / "valid.yml"
    valid_path.write_text(
        "WorkspacePreserved:\n  value: 9\n",
        encoding="utf-8",
    )
    invalid_path = tmp_path / "invalid.yml"
    invalid_path.write_text("broken: [\n", encoding="utf-8")

    load_config(valid_path)
    with pytest.raises(yaml.YAMLError):
        load_config(invalid_path)

    assert global_config.filename == "valid"
    assert create("WorkspacePreserved").value == 9
