# UserAggregation

This is a lite version of codes for our paper entitled *User-based Gradient Aggregation for Label Protection in Vertical Federated Recommendation*.

## Environment

- Python >= 3.8
- TensorFlow 2.17.0 (with CUDA support)
- CUDA 11.8 & cuDNN 8.6 (if using GPU and not relying on pip's automatic CUDA installation)
- Other dependencies are listed in `requirements.txt`


```bash
# install the dependencies
python3 -m pip install -r requirements.txt
# compile the protobuf
python3 -m grpc_tools.protoc \
    --proto_path=./protos/ \
    --python_out=./python/ \
    --grpc_python_out=./python/ \
    $(find ./protos -name "*.proto")
```

If you need GPU support, it is recommended to use `tensorflow[and-cuda]==2.17.0` in your requirements, and pip will automatically install the required CUDA dependencies.

## Dataset Preparation

Please follow these steps to prepare the dataset:

1. Download the required dataset as specified in the paper or project documentation.
2. Place the dataset files in the `data/` directory (create the directory if it does not exist).
3. Update the configuration files or scripts to point to the correct dataset path if necessary.

To preprocess the dataset, you can use the following command (replace `[criteo|avazu|movielens_1m]` with your dataset name):

```bash
cd data
python3 preprocess.py [kuairec|ad_click|movielens_1m]  # it may take a while
```

> For more details on dataset format and usage, please refer to the documentation or comments in the code.

## Experiments

You can launch the training tasks using the `run.sh` script, which takes the following arguments (note the new argument order):

- **party**: one of `"host"` (collaborator) or `"guest"` (label-holder).
- **dataset**: one of `"kuairec"`, `"ad_click"`, or `"movielens_1m"`.
- **pack-size**: an integer specifying the packing size (e.g., 4, 8, 16).
- **perturb**: one of `"proj"` (ProjPert-opt), `"iso-proj"` (ProjPert-iso), `"marvell"` (Marvell), or `"none"` (NoDefense).
- **perturb param**: for `"proj"` and `"iso-proj"`, this is the sumKL threshold; for `"marvell"`, this is the constraint of perturbation; for `"none"`, this is optional.

For example, to train on the Criteo dataset with ProjPert-iso:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh host kuairec 4 iso-proj 4.0
CUDA_VISIBLE_DEVICES=0 bash run.sh guest kuairec 4 iso-proj 4.0
```

Or with Marvell:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh host kuairec 4 marvell 4.0
CUDA_VISIBLE_DEVICES=0 bash run.sh guest kuairec 4 marvell 4.0
```

Or without perturbation:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh host kuairec 4 none
CUDA_VISIBLE_DEVICES=0 bash run.sh guest kuairec 4 none
```