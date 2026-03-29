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

This repository uses "The Adventures of Sherlock Holmes" from Project Gutenberg as the custom dataset for Task 1.

### Dataset
- Source: Project Gutenberg
- Book: The Adventures of Sherlock Holmes
- Plain text URL: `https://www.gutenberg.org/cache/epub/1661/pg1661.txt`

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

Data preprocessing
I modified scripts/download_data.sh so that it:
- downloads the Sherlock Holmes text
- stores the raw file under data/sherlock/raw/
- cleans the raw text with scripts/preprocess_raw.py
- tokenizes and sentence-tokenizes the text with scripts/preprocess.py
- limits the vocabulary size to 5000
- splits the processed corpus into:
	data/sherlock/train.txt
	data/sherlock/valid.txt
	data/sherlock/test.txt
	
Run preprocessing from the repository root:
bash scripts/download_data.sh

Final corpus statistics
The final preprocessed Sherlock corpus contains:
- 4037 training segments
- 400 validation segments
- 400 test segments
Total: 4837 preprocessed segments

Training
Training was run from:
cd tools/pytorch-examples/word_language_model
Command used:
python main.py --data ../../../data/sherlock --model LSTM --epochs 10 --dropout 0.2 --save sherlock_model.pt

Note: --accel was not used because it caused a local environment error related to torch.accelerator, so training was done on CPU.

Task 1 results
Final training run:
final validation perplexity: 84.50
final test perplexity: 72.06

Text generation
Generated sample text with:
python generate.py --data ../../../data/sherlock --checkpoint sherlock_model.pt --words 200 --outf sherlock_generated.txt

Notes on the generated text
The generated text shows that the model learned parts of the style of the Sherlock Holmes corpus, including narrative prose structure and some dialogue-like phrasing. However, the output is not fully coherent across sentences, and <unk> tokens still appear because the vocabulary was capped at 5000 during preprocessing.

Files changed
For Task 1, the main changes are:
- scripts/download_data.sh
- README.md
