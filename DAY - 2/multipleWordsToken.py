import tiktoken

print("Loading tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")
print("Tokenizer loaded!\n")

long_word = "unbelievable"
tokens = tokenizer.encode(long_word)


print("=" * 50)
print(f"Word: {long_word}")
print(f"Characters: {len(long_word)}")
print(f"Number of tokens: {len(tokens)}")
print("=" * 50)


# How its 3 token?

# Show how it's broken
for i, token in enumerate(tokens):
    decoded_piece = tokenizer.decode([token])
    print(f"Token {i+1}: {token} → '{decoded_piece}'")
print()
