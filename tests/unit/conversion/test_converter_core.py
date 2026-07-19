import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest
import torch

from ppdet_pytorch.conversion import converter as converter_module
from ppdet_pytorch.conversion.converter import WeightConverter
from ppdet_pytorch.conversion.models import (
    CheckpointFile,
    CheckpointFormat,
    ConversionConfig,
    ConversionSession,
    ConversionStatistics,
    ConversionStatus,
    Framework,
    MappingType,
    ParameterMapping,
)


def _mapping(source: str, target: str) -> ParameterMapping:
    return ParameterMapping(
        source_name=source,
        target_name=target,
        mapping_type=MappingType.IDENTITY,
        confidence_score=1.0,
        shape_compatible=True,
    )


def test_load_checkpoint_records_portable_metadata_and_suffix_warning(
    monkeypatch, tmp_path, caplog
):
    payload = b"checkpoint bytes"
    checkpoint_path = tmp_path / "source.bin"
    checkpoint_path.write_bytes(payload)
    paddle_module = ModuleType("paddle")
    paddle_module.load = lambda path: {"weight": [1.0]}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddle", paddle_module)

    with caplog.at_level(logging.WARNING):
        state_dict, checkpoint = WeightConverter().load_paddle_checkpoint(
            str(checkpoint_path)
        )

    assert state_dict == {"weight": [1.0]}
    assert checkpoint.format == CheckpointFormat.PDPARAMS
    assert checkpoint.framework == Framework.PADDLEPADDLE
    assert checkpoint.file_size_bytes == len(payload)
    assert checkpoint.checksum == hashlib.sha256(payload).hexdigest()
    assert "Expected .pdparams file" in caplog.text


def test_load_checkpoint_rejects_non_mapping_payload(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "source.pdparams"
    checkpoint_path.write_bytes(b"invalid")
    paddle_module = ModuleType("paddle")
    paddle_module.load = lambda path: ["not", "a", "mapping"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddle", paddle_module)

    with pytest.raises(ValueError, match="Expected state dict"):
        WeightConverter().load_paddle_checkpoint(str(checkpoint_path))


def test_convert_tensor_validates_shape_and_forwards_transpose(monkeypatch):
    calls = []
    monkeypatch.setattr(
        converter_module,
        "convert_paddle_to_torch_tensor",
        lambda value, name, transpose: (
            calls.append((value, name, transpose)) or torch.ones(2, 2)
        ),
    )
    monkeypatch.setattr(
        converter_module,
        "validate_tensor_shape",
        lambda *args, **kwargs: False,
    )
    converter = WeightConverter(ConversionConfig(strict_mode=False))

    result = converter.convert_tensor(
        object(), "linear.weight", expected_shape=(3, 2), transpose=True
    )

    assert result is None
    assert calls[0][1:] == ("linear.weight", True)


@pytest.mark.parametrize("strict", [False, True])
def test_convert_tensor_handles_conversion_errors_by_mode(monkeypatch, strict):
    def fail_conversion(*args, **kwargs):
        raise RuntimeError("unsupported tensor")

    monkeypatch.setattr(
        converter_module, "convert_paddle_to_torch_tensor", fail_conversion
    )
    converter = WeightConverter(ConversionConfig(strict_mode=strict))

    if strict:
        with pytest.raises(ValueError, match="unsupported tensor"):
            converter.convert_tensor(object(), "broken")
    else:
        assert converter.convert_tensor(object(), "broken") is None


def test_convert_state_dict_tracks_shape_transpose_skip_and_missing(monkeypatch):
    converter = WeightConverter(ConversionConfig(strict_mode=False))
    calls = []

    def convert_tensor(value, source, expected_shape, transpose):
        calls.append((source, expected_shape, transpose))
        return None if source == "skip" else torch.as_tensor(value)

    monkeypatch.setattr(converter, "convert_tensor", convert_tensor)
    mappings = [
        _mapping("convert", "target.convert"),
        _mapping("skip", "target.skip"),
        _mapping("missing", "target.missing"),
    ]

    converted, statistics = converter.convert_state_dict(
        {"convert": [1.0, 2.0], "skip": [3.0]},
        target_state_dict={"target.convert": torch.zeros(2), "target.skip": (1,)},
        mappings=mappings,
        transpose_target_keys={"target.convert"},
    )

    assert torch.equal(converted["target.convert"], torch.tensor([1.0, 2.0]))
    assert statistics.total_parameters == 3
    assert statistics.converted_count == 1
    assert statistics.skipped_count == 1
    assert statistics.failed_count == 1
    assert calls == [("convert", (2,), True), ("skip", (1,), False)]


def test_memory_efficient_state_conversion_releases_batches(monkeypatch):
    converter = WeightConverter(
        ConversionConfig(memory_efficient_mode=True, batch_size=1)
    )
    source = {"first": [1.0], "second": [2.0]}
    collections = []
    monkeypatch.setattr(
        converter,
        "convert_tensor",
        lambda value, *args, **kwargs: torch.as_tensor(value),
    )
    monkeypatch.setattr(
        converter_module.gc, "collect", lambda: collections.append(True)
    )

    converted, statistics = converter.convert_state_dict(source)

    assert source == {}
    assert set(converted) == {"first", "second"}
    assert statistics.converted_count == 2
    assert len(collections) == 2


def test_convert_orchestrates_metadata_mapping_and_session(monkeypatch, tmp_path):
    mapping_path = tmp_path / "mapping.json"
    config = ConversionConfig(
        manual_mapping_file=str(tmp_path / "manual.json"),
        export_mapping=True,
        export_mapping_path=str(mapping_path),
        memory_efficient_mode=True,
        batch_size=2,
        output_metadata={"target_config": "model.yml"},
    )
    converter = WeightConverter(config)
    source_checkpoint = CheckpointFile(
        file_path="source.pdparams",
        format=CheckpointFormat.PDPARAMS,
        file_size_bytes=123,
        framework=Framework.PADDLEPADDLE,
        checksum="a" * 64,
    )
    target_checkpoint = CheckpointFile(
        file_path="target.pth",
        format=CheckpointFormat.PTH,
        file_size_bytes=456,
        framework=Framework.PYTORCH,
        checksum="b" * 64,
    )
    statistics = ConversionStatistics(total_parameters=1, converted_count=1)
    metadata = {}
    events = []
    monkeypatch.setattr(
        converter,
        "load_paddle_checkpoint",
        lambda path: ({"source": object()}, source_checkpoint),
    )
    monkeypatch.setattr(
        converter.name_mapper,
        "apply_manual_overrides",
        lambda path: events.append(("manual", path)),
    )
    monkeypatch.setattr(
        converter.name_mapper,
        "apply_naming_rules",
        lambda source, target: [_mapping("source", "target")],
    )
    monkeypatch.setattr(
        converter.name_mapper,
        "find_unmapped_keys",
        lambda source, target, mappings: (["unused_source"], ["unused_target"]),
    )
    monkeypatch.setattr(
        converter,
        "convert_state_dict",
        lambda *args: ({"target": torch.ones(1)}, statistics),
    )

    def save_checkpoint(state_dict, output_path, checkpoint_metadata):
        metadata.update(checkpoint_metadata)
        return target_checkpoint

    monkeypatch.setattr(converter, "save_torch_checkpoint", save_checkpoint)
    monkeypatch.setattr(
        converter.name_mapper,
        "export_to_json",
        lambda *args: events.append(("export", args)),
    )

    session = converter.convert(
        "source.pdparams",
        "target.pth",
        target_model_state_dict={"target": torch.zeros(1)},
        transpose_target_keys={"target"},
    )

    assert session.status == ConversionStatus.COMPLETED
    assert session.source_checkpoint is source_checkpoint
    assert session.target_checkpoint is target_checkpoint
    assert session.statistics.unmapped_source_keys == ["unused_source"]
    assert session.statistics.unmapped_target_keys == ["unused_target"]
    assert metadata["source_checkpoint_size_bytes"] == 123
    assert metadata["source_checkpoint_sha256"] == "a" * 64
    assert metadata["memory_efficient_mode"] is True
    assert metadata["parameter_batch_size"] == 2
    assert metadata["target_config"] == "model.yml"
    assert events[0] == ("manual", config.manual_mapping_file)
    assert events[1][0] == "export"


def test_convert_records_failed_session(monkeypatch):
    converter = WeightConverter()
    monkeypatch.setattr(
        converter,
        "load_paddle_checkpoint",
        lambda path: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        converter.convert("source.pdparams", "target.pth")

    assert converter.session is not None
    assert converter.session.status == ConversionStatus.FAILED
    assert converter.session.errors == ["load failed"]
    assert converter.session.end_time is not None


def test_batch_refuses_existing_mapping_without_force(tmp_path):
    input_path = tmp_path / "model.pdparams"
    input_path.write_bytes(b"source")
    mapping_directory = tmp_path / "mappings"
    mapping_directory.mkdir()
    mapping_path = mapping_directory / "model.mapping.json"
    mapping_path.write_text("existing", encoding="utf-8")

    summary = WeightConverter().convert_batch(
        [str(input_path)],
        str(tmp_path / "outputs"),
        mapping_directory=str(mapping_directory),
    )

    assert summary.failed_count == 1
    assert summary.results[0].error == "Mapping file already exists"
    assert mapping_path.read_text(encoding="utf-8") == "existing"


def test_batch_isolates_failure_and_cleans_only_new_artifacts(monkeypatch, tmp_path):
    failed_input = tmp_path / "failed.pdparams"
    passed_input = tmp_path / "passed.pdparams"
    failed_input.write_bytes(b"failed")
    passed_input.write_bytes(b"passed")
    output_directory = tmp_path / "outputs"
    mapping_directory = tmp_path / "mappings"

    def convert(self, input_path, output_path, **kwargs):
        Path(output_path).write_bytes(b"checkpoint")
        if self.config.export_mapping_path:
            Path(self.config.export_mapping_path).write_text(
                "mapping", encoding="utf-8"
            )
        self.session = ConversionSession(config=self.config)
        if Path(input_path).stem == "failed":
            self.session.add_error("conversion failed")
            raise RuntimeError("conversion failed")
        self.session.status = ConversionStatus.COMPLETED
        self.session.end_time = datetime.now()
        self.session.statistics.converted_count = 2
        return self.session

    monkeypatch.setattr(WeightConverter, "convert", convert)

    summary = WeightConverter().convert_batch(
        [str(failed_input), str(passed_input)],
        str(output_directory),
        mapping_directory=str(mapping_directory),
    )

    assert summary.failed_count == 1
    assert summary.succeeded_count == 1
    assert not (output_directory / "failed.pth").exists()
    assert not (mapping_directory / "failed.mapping.json").exists()
    assert (output_directory / "passed.pth").is_file()
    assert (mapping_directory / "passed.mapping.json").is_file()
