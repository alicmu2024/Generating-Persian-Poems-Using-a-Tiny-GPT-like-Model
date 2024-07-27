import torch
import argparse
import sentencepiece as spm
from tqdm import tqdm
from dataset import load_data, CharTokenizer, WordTokenizer, SubwordTokenizer, get_batch
from model import GPTLanguageModel, post_process_text
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

# Argument parser for user input
parser = argparse.ArgumentParser(description='Train a GPT-like language model.')
parser.add_argument('--combined_file', type=str, required=True,
                    help='Path to the combined text file.')
parser.add_argument('--tokenizer', type=str, choices=['char', 'word', 'subword'], default='subword',
                    help='Choose the tokenizer: "char" for character-level, "word" for word-level, or "subword" for subword-level.')
parser.add_argument('--max_vocab_size', type=int, default=2000,
                    help='Maximum vocabulary size for word tokenizer or subword tokenizer.')
parser.add_argument('--model_prefix', type=str, default='sp_model',
                    help='Model prefix for SentencePiece tokenizer (required for subword tokenizer).')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--block_size', type=int, default=256,
                    help='Block size for input sequences.')
parser.add_argument('--learning_rate', type=float, default=2e-4,
                    help='Learning rate for the optimizer.')
parser.add_argument('--max_iters', type=int, default=2000,
                    help='Maximum number of training iterations.')
parser.add_argument('--eval_iters', type=int, default=100,
                    help='Number of iterations for evaluation on the validation set.')
parser.add_argument('--eval_interval', type=int, default=500,
                    help='Number of iterations between evaluations on the validation set.')
parser.add_argument('--patience', type=int, default=5,
                    help='Patience for early stopping.')
parser.add_argument('--accumulation_steps', type=int, default=4,
                    help='Number of steps for gradient accumulation.')
parser.add_argument('--scheduler', type=str, choices=['constant', 'step', 'exponential', 'cosine'], default='step',
                    help='Type of learning rate scheduler: "constant", "step", "exponential", or "cosine".')
parser.add_argument('--step_size', type=int, default=500,
                    help='Step size for StepLR scheduler.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Multiplicative factor for learning rate decay (for StepLR and ExponentialLR).')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for text generation.')
parser.add_argument('--pretrained_weights', type=str, default=None,
                    help='Path to the pretrained weights file (optional).')
parser.add_argument('--top_k', type=int, default=50,
                    help='Top-k sampling parameter for text generation.')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p (nucleus) sampling parameter for text generation.')
parser.add_argument('--min_line_length', type=int, default=10,
                    help='Minimum line length for post-processing generated text.')

args = parser.parse_args()

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data based on the selected tokenizer
if args.tokenizer == 'subword':
    import os
    if not os.path.exists(f"{args.model_prefix}.model"):
        raise FileNotFoundError(f"SentencePiece model '{args.model_prefix}.model' not found. Please train the model first.")
    tokenizer_class = SubwordTokenizer
    tokenizer_kwargs = {'model_prefix': args.model_prefix}
elif args.tokenizer == 'char':
    tokenizer_class = CharTokenizer
    tokenizer_kwargs = {'text': open(args.combined_file, 'r', encoding='utf-8').read()}
else:
    tokenizer_class = WordTokenizer
    tokenizer_kwargs = {'text': open(args.combined_file, 'r', encoding='utf-8').read(), 'max_vocab_size': args.max_vocab_size}

train_data, val_data, vocab_size, encode, decode = load_data(args.combined_file, tokenizer_class, **tokenizer_kwargs)
print(f"Vocabulary size: {vocab_size}")

# Create model
model = GPTLanguageModel(vocab_size).to(device)

# Load pretrained weights if provided
if args.pretrained_weights:
    model.load_state_dict(torch.load(args.pretrained_weights))
    print(f"Loaded pretrained weights from {args.pretrained_weights}")

# Print the number of parameters in the model
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# Create the learning rate scheduler based on user input
if args.scheduler == 'constant':
    scheduler = None  # No scheduler, constant learning rate
elif args.scheduler == 'step':
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
elif args.scheduler == 'exponential':
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
elif args.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iters)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler()

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(train_data, val_data, args.batch_size, args.block_size, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

best_val_loss = float('inf')
patience_counter = 0

for iter in tqdm(range(args.max_iters), desc="Training Progress"):
    # Evaluate the loss on train and val sets
    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping logic
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            # Optionally save the model
            torch.save(model.state_dict(), 'PATH TO THE PRE-TRAINED MODEL.')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # Gradient accumulation
    for _ in range(args.accumulation_steps):
        xb, yb = get_batch(train_data, val_data, args.batch_size, args.block_size, 'train')
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            logits, loss = model(xb, yb)

        loss = loss / args.accumulation_steps  # Normalize the loss
        scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()  # Update the learning rate if a scheduler is used

# Generate and post-process sample text
starting_word = "محبت"  # Replace with your desired Persian word
max_new_tokens = 400  # Specify the number of tokens to generate

starting_tokens = encode(starting_word)
context = torch.tensor(starting_tokens, dtype=torch.long, device=device).unsqueeze(0)

generated_tokens = model.generate(context,
                                  max_new_tokens=max_new_tokens,
                                  temperature=args.temperature,
                                  top_k=args.top_k,
                                  top_p=args.top_p)

generated_text = decode(generated_tokens[0].tolist())
processed_text = post_process_text(generated_text, min_line_length=args.min_line_length)

print("Generated and processed text:")
print(processed_text)

print("Training completed.")
