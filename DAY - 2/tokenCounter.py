# AUTHOR: Joel Siby
# DATE: 29 / 03 / 2026
# PURPOSE: Understand tokens, context windows, and why RAG exists

# 1. What Are Tokens?
# LLMs (like GPT, Llama, Mistral) do NOT see text as words. They see text as tokens.
# One token is NOT one word

# Common words = 1 token	"the", "a", "I", "cat"
# Less common words = multiple tokens	"unbelievable" = "un" + "believe" + "able"
# Punctuation = separate tokens	"." = token_13
# Spaces sometimes = tokens	" " = token_220

import tiktoken

# We'll use cl100k_base (the tokenizer for GPT-4)
# This is the industry standard for understanding token counts
print("Loading tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")
print("Tokenizer loaded!\n")

print("=" * 50)
print("SECTION 2: Tokens are NOT words")
print("=" * 50)


# Simple sentence
sentence = "The cat sat on the mat"
tokens = tokenizer.encode(sentence)

print(f"Sentence: {sentence}")
print(f"Number of words: {len(sentence.split())}")
print(f"Number of tokens: {len(tokens)}")
print(f"Tokens: {tokens}")
print()
print("=" * 50)

# Decode tokens back to text (shows what the model "sees")
decoded = tokenizer.decode(tokens)
print(f"Decoded back: {decoded}")
print()
