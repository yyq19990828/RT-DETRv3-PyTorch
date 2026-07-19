import json
from pathlib import Path

import pytest

from ppdet_pytorch.cli.convert import (
    build_target_state_dict,
    create_argument_parser,
    discover_input_paths,
    main,
    validate_arguments,
)
from ppdet_pytorch.conversion.models import (
    BatchConversionResult,
    BatchConversionSummary,
    ConversionStatus,
)

ROOT = Path(__file__).resolve().parents[3]
R18_CONFIG = ROOT / "configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml"


def _parse_args(tmp_path, *extra_args):
    input_path = tmp_path / "input.pdparams"
    input_path.write_bytes(b"fixture")
    return create_argument_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.pth"),
            *extra_args,
        ]
    )


def test_validation_requires_target_config_by_default(tmp_path):
    args = _parse_args(tmp_path)

    with pytest.raises(SystemExit) as error:
        validate_arguments(args)

    assert error.value.code == 1


def test_no_validate_explicitly_allows_missing_target_config(tmp_path):
    args = _parse_args(tmp_path, "--no-validate")

    validate_arguments(args)


def test_validation_accepts_existing_yaml_target_config(tmp_path):
    args = _parse_args(tmp_path, "--config", str(R18_CONFIG))

    validate_arguments(args)


def test_build_target_state_dict_from_r18_config(isolated_workspace):
    state_dict, architecture, transpose_target_keys = build_target_state_dict(
        str(R18_CONFIG)
    )

    assert architecture == "RTDETRV3"
    assert len(state_dict) == 648
    assert state_dict["backbone.conv1.conv1_1.conv.weight"] == (32, 3, 3, 3)
    assert "backbone.conv1.conv1_1.conv.weight" in state_dict
    assert "transformer.enc_output.0.0.weight" in transpose_target_keys
    assert "transformer.map_memory.0.weight" in transpose_target_keys
    assert "transformer.denoising_class_embed.weight" not in transpose_target_keys


def test_batch_discovers_directory_inputs_in_stable_order(tmp_path):
    (tmp_path / "b.pdparams").write_bytes(b"b")
    (tmp_path / "a.pdparams").write_bytes(b"a")
    (tmp_path / "ignored.txt").write_bytes(b"ignored")

    discovered = discover_input_paths(str(tmp_path))

    assert [path.name for path in discovered] == ["a.pdparams", "b.pdparams"]


def test_batch_validation_accepts_input_directory(tmp_path):
    input_directory = tmp_path / "inputs"
    input_directory.mkdir()
    (input_directory / "model.pdparams").write_bytes(b"fixture")
    args = create_argument_parser().parse_args(
        [
            "--batch",
            "--input",
            str(input_directory),
            "--output",
            str(tmp_path / "outputs"),
            "--no-validate",
            "--memory-efficient",
            "--parameter-batch-size",
            "8",
        ]
    )

    validate_arguments(args)


def test_parameter_batch_size_must_be_positive(tmp_path):
    args = _parse_args(
        tmp_path,
        "--no-validate",
        "--memory-efficient",
        "--parameter-batch-size",
        "0",
    )

    with pytest.raises(SystemExit) as error:
        validate_arguments(args)

    assert error.value.code == 1


def test_batch_cli_writes_summary(monkeypatch, tmp_path):
    input_path = tmp_path / "model.pdparams"
    input_path.write_bytes(b"fixture")
    output_directory = tmp_path / "outputs"
    summary_path = tmp_path / "summary.json"
    summary = BatchConversionSummary(output_directory=str(output_directory))
    summary.results.append(
        BatchConversionResult(
            source_path=str(input_path),
            output_path=str(output_directory / "model.pth"),
            status=ConversionStatus.COMPLETED,
            converted_count=2,
        )
    )
    summary.finish()
    monkeypatch.setattr(
        "ppdet_pytorch.cli.convert.WeightConverter.convert_batch",
        lambda *args, **kwargs: summary,
    )
    argv = [
        "--batch",
        "--input",
        str(input_path),
        "--output",
        str(output_directory),
        "--summary",
        str(summary_path),
        "--no-validate",
    ]

    with pytest.raises(SystemExit) as error:
        main(argv)

    assert error.value.code == 0
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["succeeded_count"] == 1
    assert summary_payload["failed_count"] == 0
