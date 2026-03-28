# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marcamsler1/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh


## Task 1: Custom dataset

This repository uses **Alice’s Adventures in Wonderland** from Project Gutenberg as the custom dataset for Task 1.

### Dataset
- Source: Project Gutenberg
- Book: *Alice’s Adventures in Wonderland*
- Plain text URL: `https://www.gutenberg.org/cache/epub/11/pg11.txt`

### Environment setup
Create and activate the virtual environment:

```bash
python3.11 -m virtualenv venvs/torch3
source venvs/torch3/bin/activate

# Requirements
Install the required packages:
./scripts/install_packages.sh
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('punkt')"

# Steps

Preprocessing:
The script scripts/download_data.sh was adapted to:
- download the Alice text
- clean the raw text
- tokenize and sentence-tokenize it
- limit the vocabulary size to 5000
- split the processed text into:
data/alice/train.txt
data/alice/valid.txt
data/alice/test.txt

To download and preprocess the dataset, run from the repository root:
bash scripts/download_data.sh

Training: 
From tools/pytorch-examples/word_language_model, train the Task 1 model with:
python main.py --data ../../../data/alice --model LSTM --epochs 10 --dropout 0.2 --save alice_model.pt
--accel was not used because it caused an error related to torch.accelerator in the local environment.

Text generation:
From tools/pytorch-examples/word_language_model, generate sample text with:
python generate.py --data ../../../data/alice --checkpoint alice_model.pt --outf alice_sample.txt --words 200

# Results of task 1
Task 1 results
Final validation perplexity: 162.92
Final test perplexity: 122.71



