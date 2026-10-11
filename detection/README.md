## Introduction

Real-time object detection training, inference, serving.


## Usage
1. Build docker images
    ```bash
    docker compose build
    ```

2. start service
    ```bash
    docker compose  up -d
    ```

+ send request reference `tests/integration/test_serve.py`


## Details
+ dataset file architecture
    + dataset/
        + images/
            + raw/
                + *<dataset_name>*/ ------> can be nested folders with jpg, jpeg, png
            + remove_exif/
                + *<dataset_name>*/ ------> same file structure as above
        + labels/
            + detection/
                + *<exp_name>*/
                    + instances_default.json ------> labeled format by CVAT coco
            + detection_transform/
                + *<exp_name>*/
                    + .others
                        + coco.json ------> convert instances_default.json to coco by adding full path
                        + general.json ------> convert from coco.json, for show_labels.py 
                        + general_train.json ------> for cosmodules postprocessing
                        + general_valid.json ------> for cosmodules postprocessing
                    + train/
                        + _annotations.coco.json ------> detection training need
                    + valid/
                        + _annotations.coco.json ------> detection training need

1. Data preparation: `scripts/dataset/*.py`
    1. remove_exif, convert `images/raw/` to `images/remove_exif/`
    2. **manual** label to `labels/detection/`
    3. convert to `.others/coco.json` by adding full path and `.others/general.json`
    4. show labels to `outputs/show_labels/`
    5. split label to `train/_annotations.coco.json` and `valid/_annotations.coco.json`

2. training: `scripts/train/*.py`
    + train.py: output metrics saved at `outputs/training/`
    + show_history.py: plot loss and ap curves during training at `outputs/training/`

3. eval: `scripts/eval/*.py`
    + eval.py
        + inference to `outputs/training/<exp_name>/eval/predictions.json`
        + compute metrics to `outputs/training/<exp_name>/eval/`
    + visualize.py (optional)
        + plot all detection result to `outputs/training/<exp_name>/eval/vis`

4. serving
    + inference: `src/infer.py`
    + serve: `serve.py`
