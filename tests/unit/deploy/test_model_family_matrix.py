from pathlib import Path

import pytest

from ppdet_pytorch.core.workspace import create, load_config

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = (
    *(f"configs/dfine/dfine_hgnetv2_{variant}_coco.yml" for variant in "nsmlx"),
    *(f"configs/deim/dfine/deim_hgnetv2_{variant}_coco.yml" for variant in "nsmlx"),
    "configs/deim/rtdetrv2/deim_r18vd_120e_coco.yml",
    "configs/deim/rtdetrv2/deim_r34vd_120e_coco.yml",
    "configs/deim/rtdetrv2/deim_r50vd_m_60e_coco.yml",
    "configs/deim/rtdetrv2/deim_r50vd_60e_coco.yml",
    "configs/deim/rtdetrv2/deim_r101vd_60e_coco.yml",
    *(f"configs/rtdetrv4/rtdetrv4_hgnetv2_{variant}_coco.yml" for variant in "smlx"),
    *(f"configs/deimv2/deimv2_dinov3_{variant}_coco.yml" for variant in "smlx"),
    *(f"configs/deimv2/deimv2_hgnetv2_{variant}_coco.yml" for variant in ("n", "pico", "femto", "atto")),
)


@pytest.mark.parametrize("config_path", CONFIGS)
def test_all_model_family_variants_enter_student_deploy_mode(
    config_path, isolated_workspace
):
    config = load_config(ROOT / config_path)
    model = create(config.architecture).eval()

    assert model.deploy() is model
    assert not any(
        "teacher" in name or "distill" in name for name in model.state_dict()
    )
