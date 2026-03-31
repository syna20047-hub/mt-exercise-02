# MT Exercise 2: Pytorch RNN Language Models

## Task 1: Custom dataset

This repository uses *The Adventures of Sherlock Holmes* from Project Gutenberg as the custom dataset for Task 1.

### Dataset
- **Source:** Project Gutenberg
- **Book:** *The Adventures of Sherlock Holmes*
- **Plain text URL:** `https://www.gutenberg.org/cache/epub/1661/pg1661.txt`

### Environment setup
Create and activate the virtual environment:

```bash
python3.11 -m virtualenv venvs/torch3
source venvs/torch3/bin/activate
```

### Requirements

Install the required packages:

```bash
./scripts/install_packages.sh
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('punkt')"
```

### Data preprocessing

I modified `scripts/download_data.sh` so that it:

- downloads the Sherlock Holmes text
- stores the raw file under `data/sherlock/raw/`
- cleans the raw text with `scripts/preprocess_raw.py`
- tokenizes and sentence-tokenizes the text with `scripts/preprocess.py`
- limits the vocabulary size to 5000
- splits the processed corpus into:
	`data/sherlock/train.txt`
	`data/sherlock/valid.txt`
	`data/sherlock/test.txt`
	
Run preprocessing from the repository root:

```bash
bash scripts/download_data.sh
```

### Final corpus statistics
The final preprocessed Sherlock corpus contains:
- **4037** training segments
- **400** validation segments
- **400** test segments
**Total:** 4837 preprocessed segments

### Training

Training was run from:

```bash
cd tools/pytorch-examples/word_language_model
```

Command used:

```bash
python main.py --data ../../../data/sherlock --model LSTM --epochs 10 --dropout 0.2 --save sherlock_model.pt
```

> **Note:** `--accel` was not used because it caused a local environment error related to `torch.accelerator`, so training was done on CPU.

### Task 1 Results

Final training run:

- **Final validation perplexity:** 84.50
- **Final test perplexity:** 72.06

### Text Generation

Generated sample text with:

```bash
python generate.py --data ../../../data/sherlock --checkpoint sherlock_model.pt --words 200 --outf sherlock_generated.txt
```

### Notes on the Generated Text

The generated text shows that the model learned parts of the style of the Sherlock Holmes corpus, including narrative prose structure and some dialogue-like phrasing. However, the output is not fully coherent across sentences, and <unk> tokens still appear because the vocabulary was capped at 5000 during preprocessing.

### Files changed

For Task 1, the main changes are:
- `scripts/download_data.sh`
- `README.md`

## Task 2: Dropout Experiment

For Task 2, I trained 5 LSTM language models on the Sherlock Holmes dataset from Task 1, varying only the dropout setting:

- `0.0`
- `0.2`
- `0.4`
- `0.6`
- `0.8`

### Changes Made

- Modified `tools/pytorch-examples/word_language_model/main.py`
  - added a `--logfile` argument
  - saved training, validation, and final test perplexities in TSV format
- Added `tools/pytorch-examples/word_language_model/analyze_dropout_logs.py`
  - reads the TSV log files
  - creates:
    - training perplexity table
    - validation perplexity table
    - test perplexity table
    - training perplexity plot
    - validation perplexity plot

### Commands and Order of Execution

From the repository root:

```bash
source venvs/torch3/bin/activate
bash scripts/download_data.sh
cd tools/pytorch-examples/word_language_model
```

Run the 5 dropout experiments:

```bash
python main.py --data ../../../data/sherlock --model LSTM --epochs 10 --dropout 0.0 --save sherlock_d0.pt --logfile logs/log_d0.tsv
python main.py --data ../../../data/sherlock --model LSTM --epochs 10 --dropout 0.2 --save sherlock_d02.pt --logfile logs/log_d02.tsv
python main.py --data ../../../data/sherlock --model LSTM --epochs 10 --dropout 0.4 --save sherlock_d04.pt --logfile logs/log_d04.tsv
python main.py --data ../../../data/sherlock --model LSTM --epochs 10 --dropout 0.6 --save sherlock_d06.pt --logfile logs/log_d06.tsv
python main.py --data ../../../data/sherlock --model LSTM --epochs 10 --dropout 0.8 --save sherlock_d08.pt --logfile logs/log_d08.tsv
```

Create tables and plots:
```bash
python analyze_dropout_logs.py
```

Generate text from best and worst models:

```bash
python generate.py --data ../../../data/sherlock --checkpoint sherlock_d02.pt --words 200 --outf sherlock_best.txt
python generate.py --data ../../../data/sherlock --checkpoint sherlock_d08.pt --words 200 --outf sherlock_worst.txt
```

### Final test perplexities:
- dropout 0.0 -> 83.60
- dropout 0.2 -> 72.06
- dropout 0.4 -> 74.82
- dropout 0.6 -> 77.35
- dropout 0.8 -> 101.42

### Summary
- **Best model:** dropout `0.2` with test perplexity **72.06**
- **Worst model:** dropout `0.8` with test perplexity **101.42**