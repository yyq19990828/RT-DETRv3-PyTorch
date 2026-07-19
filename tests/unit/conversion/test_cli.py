import json
from pathlib import Path
from types import SimpleNamespace

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


def test_validation_rejects_invalid_input_output_and_auxiliary_paths(tmp_path):
    parser = create_argument_parser()
    input_path = tmp_path / "input.pdparams"
    input_path.write_bytes(b"fixture")
    batch_directory = tmp_path / "inputs"
    batch_directory.mkdir()
    (batch_directory / "model.pdparams").write_bytes(b"fixture")
    output_file = tmp_path / "existing.pth"
    output_file.write_bytes(b"existing")
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text("{}", encoding="utf-8")
    summary_file = tmp_path / "summary.json"
    summary_file.write_text("{}", encoding="utf-8")
    non_yaml_config = tmp_path / "config.txt"
    non_yaml_config.write_text("architecture: Model\n", encoding="utf-8")

    invalid_argv = [
        [
            "--input",
            str(tmp_path / "missing.pdparams"),
            "--output",
            str(tmp_path / "output.pth"),
            "--no-validate",
        ],
        [
            "--input",
            str(input_path),
            "--output",
            str(output_file),
            "--no-validate",
        ],
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.pth"),
            "--summary",
            str(summary_file),
            "--no-validate",
        ],
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.pth"),
            "--manual-mapping",
            str(tmp_path / "missing.json"),
            "--no-validate",
        ],
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.pth"),
            "--config",
            str(tmp_path / "missing.yml"),
        ],
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.pth"),
            "--config",
            str(non_yaml_config),
        ],
        [
            "--batch",
            "--input",
            str(tmp_path / "empty"),
            "--output",
            str(tmp_path / "outputs"),
            "--no-validate",
        ],
        [
            "--batch",
            "--input",
            str(batch_directory),
            "--output",
            str(output_file),
            "--no-validate",
        ],
        [
            "--batch",
            "--input",
            str(batch_directory),
            "--output",
            str(tmp_path / "outputs"),
            "--save-mapping",
            str(mapping_file),
            "--no-validate",
        ],
        [
            "--batch",
            "--input",
            str(batch_directory),
            "--output",
            str(tmp_path / "outputs"),
            "--summary",
            str(summary_file),
            "--no-validate",
        ],
    ]

    for argv in invalid_argv:
        with pytest.raises(SystemExit) as error:
            validate_arguments(parser.parse_args(argv))
        assert error.value.code == 1


def test_validation_warns_but_accepts_existing_nonstandard_input_suffix(tmp_path):
    input_path = tmp_path / "checkpoint.bin"
    input_path.write_bytes(b"fixture")
    args = create_argument_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.pth"),
            "--no-validate",
        ]
    )

    validate_arguments(args)


def test_batch_discovers_glob_inputs_in_stable_order(tmp_path):
    (tmp_path / "b.pdparams").write_bytes(b"b")
    (tmp_path / "a.pdparams").write_bytes(b"a")

    discovered = discover_input_paths(str(tmp_path / "*.pdparams"))

    assert [path.name for path in discovered] == ["a.pdparams", "b.pdparams"]


def test_single_cli_wires_target_aware_conversion(monkeypatch, tmp_path):
    input_path = tmp_path / "input.pdparams"
    input_path.write_bytes(b"fixture")
    config_path = tmp_path / "model.yml"
    config_path.write_text("architecture: FakeModel\n", encoding="utf-8")
    manual_mapping = tmp_path / "manual.json"
    manual_mapping.write_text("{}", encoding="utf-8")
    mapping_output = tmp_path / "mapping.json"
    observed = {}
    session = SimpleNamespace(
        session_id="session-1",
        duration_seconds=1.25,
        statistics=SimpleNamespace(
            total_parameters=3,
            converted_count=2,
            skipped_count=1,
            unmapped_source_keys=["source.extra"],
            unmapped_target_keys=["target.missing"],
        ),
        warnings=["warning"],
    )

    class FakeConverter:
        def __init__(self, config):
            observed["config"] = config

        def convert(self, **kwargs):
            observed["convert"] = kwargs
            return session

    log_levels = []
    monkeypatch.setenv("PADDLE_CONV_LOG_LEVEL", "WARNING")
    monkeypatch.setattr(
        "ppdet_pytorch.cli.convert.configure_logging", log_levels.append
    )
    monkeypatch.setattr(
        "ppdet_pytorch.cli.convert.build_target_state_dict",
        lambda path: ({"linear.weight": (2, 2)}, "FakeModel", {"linear.weight"}),
    )
    monkeypatch.setattr("ppdet_pytorch.cli.convert.WeightConverter", FakeConverter)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "output.pth"),
                "--config",
                str(config_path),
                "--manual-mapping",
                str(manual_mapping),
                "--save-mapping",
                str(mapping_output),
                "--strict",
                "--memory-efficient",
                "--parameter-batch-size",
                "8",
                "--log-level",
                "DEBUG",
            ]
        )

    assert error.value.code == 0
    assert log_levels == ["DEBUG", "WARNING"]
    conversion_config = observed["config"]
    assert conversion_config.strict_mode is True
    assert conversion_config.manual_mapping_file == str(manual_mapping)
    assert conversion_config.export_mapping_path == str(mapping_output)
    assert conversion_config.memory_efficient_mode is True
    assert conversion_config.batch_size == 8
    assert conversion_config.log_level == "WARNING"
    assert conversion_config.output_metadata == {
        "target_validation": True,
        "batch_conversion": False,
        "target_config": str(config_path),
        "target_architecture": "FakeModel",
    }
    assert observed["convert"] == {
        "input_path": str(input_path),
        "output_path": str(tmp_path / "output.pth"),
        "target_model_state_dict": {"linear.weight": (2, 2)},
        "transpose_target_keys": {"linear.weight"},
    }


@pytest.mark.parametrize(
    ("conversion_error", "exit_code"),
    [(KeyboardInterrupt(), 130), (RuntimeError("conversion failed"), 1)],
)
def test_single_cli_maps_conversion_failures_to_exit_codes(
    monkeypatch, tmp_path, conversion_error, exit_code
):
    input_path = tmp_path / "input.pdparams"
    input_path.write_bytes(b"fixture")

    class FailingConverter:
        def __init__(self, config):
            pass

        def convert(self, **kwargs):
            raise conversion_error

    monkeypatch.setattr("ppdet_pytorch.cli.convert.WeightConverter", FailingConverter)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "output.pth"),
                "--no-validate",
            ]
        )

    assert error.value.code == exit_code


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
