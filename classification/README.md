## Introduction

Vertical flip classification training, inference, serving.


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
            + crop/
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
            + classification/
                + *<exp_name>*/
                    + unlabeled.json ------> extract images from detection_transform/exp_name/.others/general.json
                    + labeled.json ------> same as above, but do manual labele on this file
            + classification_transform/ ------> classification training need, and also cosmodules
                + *<exp_name>*/
                    + train.json
                    + valid.json

+ labels/classification* are all in cosmodules general format

1. Data preparation: `scripts/preprocess/*.py`
    1. convert `.others/general.json` to `classification/`
    2. **manual** label to `labels/classification/`
    3. split label to `labels/classification_transform`

2. training
    + train.py: output metrics saved at `outputs/`
    + show_history.py: plot loss and ap curves during training at `outputs/`

3. eval
    + eval.py
        + inference to `outputs/<exp_name>/eval/predictions.json`
        + compute metrics to `outputs/<exp_name>/eval/`

4. serving
    + inference: `src/infer.py`
    + serve: `serve.py`
