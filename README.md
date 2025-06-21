# Toxic Text Classification with RoBERTa

This repository provides a pipeline to classify toxic text using a pre-trained RoBERTa model and fine-tuning it on a custom dataset.

## Contents

- `toxic_check.py`: A simple script that uses a pre-trained unbiased-toxic-roberta model to classify input text as toxic or not.
- `fine_tune.py`: A fine-tuning script that trains the RoBERTa model on your labeled dataset (CSV files with `text` and `label` columns).
- `toxic_check_fine_tuned.py`: A script to classify text using the fine-tuned model.
- Example datasets: `train.csv` and `val.csv` with text samples labeled as toxic (`1`) or not toxic (`0`).

## Requirements

### 1. System setup (Ubuntu/Debian-based)

```bash
sudo apt update
sudo apt install -y python3.12-venv
python3.12 -m venv ~/myenv
source ~/myenv/bin/activate
```

### 2. Python dependencies

```bash
pip install transformers[torch]
pip install pandas scikit-learn
```
Python 3.7+ is required. Make sure your environment is activated each time you use the scripts:
```
source ~/myenv/bin/activate
```

## How to Use

1. **Check toxicity with pre-trained model**

```bash
python3 toxic_check.py
```
Input any sentence when prompted, and it outputs the toxicity label and confidence score.

2. **Fine-tune the model**

Prepare your dataset in `train.csv` and `val.csv` with two columns: `text` and `label` (0 or 1).

Run the fine-tuning script:
```bash
python3 fine_tune.py
```
This will train the model and save it under `./finetuned-toxic-roberta`.

3. **Check toxicity with the fine-tuned model**

Run:
```bash
python3 toxic_check_fine_tuned.py
```
Input a sentence to get toxicity prediction based on your fine-tuned model.

# Dataset Format Example

```
text,label
You are worthless,1
I hate everything about you,1
Get out of here now,1
You make me sick,1
```

# Notes

- Labels: `1` means toxic, `0` means not toxic.
- The fine-tuning script uses binary classification.
- Make sure your dataset text does not contain commas, or adjust CSV quoting accordingly.

