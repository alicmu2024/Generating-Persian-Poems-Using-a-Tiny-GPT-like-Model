import torch
import sentencepiece as spm
from collections import Counter
import re

class Tokenizer:
    def encode(self, text):
        raise NotImplementedError("Encode method not implemented.")

    def decode(self, tokens):
        raise NotImplementedError("Decode method not implemented.")

    def get_vocab_size(self):
        raise NotImplementedError("get_vocab_size method not implemented.")

class CharTokenizer(Tokenizer):
    def __init__(self, text, special_tokens=['<BOS>', '<EOS>']):
        # Add special tokens to the set of characters
        self.chars = sorted(list(set(text ))) # + ''.join(special_tokens))))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

    def get_vocab_size(self):
        return len(self.chars)

class WordTokenizer(Tokenizer):
    def __init__(self, text, max_vocab_size=4000, special_tokens=['<BOS>', '<EOS>']):
        words = text.split()
        word_freq = Counter(words)
        self.words = [word for word, _ in word_freq.most_common(max_vocab_size)] # - len(special_tokens))]
        self.stoi = {word: i for i, word in enumerate(self.words)}
        self.itos = {i: word for i, word in enumerate(self.words)}

    def encode(self, text):
        return [self.stoi.get(word, 0) for word in text.split()]  # 0 is the index for <unk>

    def decode(self, tokens):
        return ' '.join([self.itos[i] for i in tokens])

    def get_vocab_size(self):
        return len(self.words)

class SubwordTokenizer(Tokenizer):
    def __init__(self, model_prefix):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{model_prefix}.model')
        print(f"Loaded SentencePiece model with vocabulary size: {self.sp.get_piece_size()}")

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, tokens):
        return self.sp.decode_ids(tokens)

    def get_vocab_size(self):
        return self.sp.get_piece_size()

def normalize_persian_text(text):
    # Normalize Persian characters
    # text = text.replace('ي', 'ی').replace('ك', 'ک')
    # Normalize spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_empty_lines(text):
    # Split the text into lines and remove empty lines
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(non_empty_lines)

# def add_bos_eos_tokens(text, bos_token='<BOS>', eos_token='<EOS>'):
#     sentences = text.split('\n')
#     processed_sentences = [f"{bos_token} {normalize_persian_text(sentence)} {eos_token}" for sentence in sentences if sentence.strip()]
#     return ' '.join(processed_sentences)

def load_data(file_path, tokenizer_class, **tokenizer_kwargs):
    # Read the combined text file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove empty lines
    text = remove_empty_lines(text)
    # Add BOS and EOS tokens
    # text = add_bos_eos_tokens(text)

    # Create the tokenizer
    if tokenizer_class in [CharTokenizer, WordTokenizer]:
        # Remove 'text' from tokenizer_kwargs if it's there
        tokenizer_kwargs.pop('text', None)
        tokenizer = tokenizer_class(text, **tokenizer_kwargs)
    else:
        tokenizer = tokenizer_class(**tokenizer_kwargs)

    # Encode data
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # First 90% for train, rest for validation
    train_data = data[:n]
    val_data = data[n:]

    # Print the number of tokens in the training and validation datasets
    print(f"Number of tokens for training: {len(train_data)}")
    print(f"Number of tokens for validation: {len(val_data)}")
    # print(f"Vocabulary size: {tokenizer.get_vocab_size()}")  # Print the vocabulary size

    return train_data, val_data, tokenizer.get_vocab_size(), tokenizer.encode, tokenizer.decode

def get_batch(train_data, val_data, batch_size, block_size, split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to('cuda' if torch.cuda.is_available() else 'cpu'), y.to('cuda' if torch.cuda.is_available() else 'cpu')
    return x, y