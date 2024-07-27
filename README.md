# Generating-Persian-Poems-Using-a-Tiny-GPT-like-Model

## Overview

The Persian Poem Generator is a machine learning project that utilizes a GPT-like language model to generate coherent and creative Persian poetry. By leveraging advanced natural language processing techniques and a character-level tokenizer, this project aims to produce poetic verses that reflect the beauty and intricacies of the Persian language.

## Features

### Character-Level Tokenization: 
The model employs a character-level tokenizer, allowing it to capture the nuances of the Persian script and generate text character by character.

### Flexible Text Generation: 
Users can specify a starting word and adjust parameters such as temperature, top-k, and top-p sampling to influence the creativity and randomness of the generated poetry.

### Post-Processing: 
The generated text undergoes post-processing to ensure coherence, proper formatting, and adherence to poetic structures.

### Interactive Interface: 
The project is designed to be user-friendly, enabling users to easily generate poetry through a simple command-line interface or Jupyter notebook.

## Installation
To set up the Persian Poem Generator, follow these steps:
Clone the Repository:
bash
git clone git@github.com:alicmu2024/Generating-Persian-Poems-Using-a-Tiny-GPT-like-Model.git

### Install Required Libraries:

Make sure you have Python and the required libraries installed. You can install the necessary packages using pip:
bash
pip install sentencepiece # for using sub-word tokenizer

## Usage

To generate Persian poetry, you can run the main.py script or use the provided Jupyter notebook. Here’s a quick guide on how to use the generator:
Command-Line Interface
Run the Training Script:
## Train the model using your dataset:
bash
python main.py --combined_file ./combined_data.txt --tokenizer char --max_vocab_size 41 --batch_size 128 --learning_rate 1e-4 --max_iters 2500 --eval_interval 200 --patience 300 --eval_iters 50 --accumulation_steps 4 --scheduler constant --step_size 500 --gamma 0.1 --temperature 1.0 --top_k 50 --top_p 0.95 --min_line_length 20
