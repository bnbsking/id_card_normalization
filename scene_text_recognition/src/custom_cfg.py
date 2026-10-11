from typing import Dict


def recursive_merge(cfg: Dict, update_cfg: Dict) -> Dict:
    for key, value in update_cfg.items():
        if key in cfg and isinstance(value, dict) and isinstance(cfg[key], dict):
            cfg[key] = cfg[key] | recursive_merge(cfg[key], value)
        else:
            cfg[key] = value
    return cfg
