from pathlib import Path

import yaml


def load_config(config_path="configs/config.yaml"):
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 형식이 올바르지 않습니다.")

    required_sections = [
        "paths",
        "dataset",
        "model",
        "inference",
        "labels",
        "recording",
        "discord",
    ]

    for section in required_sections:
        if section not in cfg:
            raise KeyError(
                f"config.yaml에 '{section}' 항목이 없습니다."
            )

    return cfg