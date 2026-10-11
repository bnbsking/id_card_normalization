from tempfile import TemporaryDirectory
from typing import List

from tools.train_rec import main_custom
import yaml

from src.custom_cfg import recursive_merge


def train(
        output_dir: str,
        train_data_dir_list: List[str],
        eval_data_dir_list: List[str],
    ):
    update_cfg = {
        "Global": {"output_dir": output_dir},
        "Train": {
            "dataset": {"data_dir_list": train_data_dir_list}
        },
        "Eval": {
            "dataset": {"data_dir_list": eval_data_dir_list}
        }
    }
    with open('/app/cfgs/train.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = recursive_merge(cfg, update_cfg)
    
    with TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/train.yaml"
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f)
        main_custom(save_path)
