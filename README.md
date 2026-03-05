# ASLAN - experiments
This repository contains the experimental pipeline for the ASLAN project (Assisted Scoring of Learner Answers through Normalization).
## Installation

We use [Poetry](https://python-poetry.org/) for dependency management. 

### 1. Prerequisites
Ensure you have **Python >=3.10** and **Poetry** installed. 

*Note: If you are using a GPU, ensure your CUDA drivers are version 11.8 or higher.*

### 2. Install Dependencies
Run the following command in the project root to create a virtual environment and install all required packages:
```bash
poetry install
```

## Usage

### Instance-based Scoring
The main entry point for the training and evaluation pipeline is [scoring_pipeline.py](scoring_pipeline.py).

When running this pipeline, you can customize the experiment using the following arguments:

| Argument | Type | Required | Default | Description                                                                                       |
| :--- | :--- | :--- | :--- |:--------------------------------------------------------------------------------------------------|
| `--dataset` | `str` | **Yes** | - | Name of the dataset (e.g., `ASAP`, alice).                                                        |
| `--experiment_name` | `str` | **Yes** | - | Unique identifier for experiment results.                                                         |
| `--train` | `str` | **Yes** | - | Path to the training data file.                                                                   |
| `--test` | `str` | **Yes** | - | Path to the test data file.                                                                       |
| `--experiment_setting` | `str` | No | `None` | Optional string to define if the question and scoring rubrics will be added to the trainind data. |

#### Example Usage on ASAP prompt 1
```bash
poetry run python3 scoring_pipeline.py \
  --dataset ASAP \
  --experiment_name instance_based_scoring_asap_1 \
  --train 1_GOLD_train.csv \
  --test 1_GOLD_test.csv \

