print("=" * 50)
print("SECTION 5: Context window limitations")
print("=" * 50)

# List of common models and their context windows
models = [
    ("GPT-3.5", 4096),
    ("GPT-4", 8192),
    ("GPT-4 Turbo", 128000),
    ("Llama 3 (8B)", 8192),
    ("Llama 3 (70B)", 128000),
    ("Claude 3", 200000),
]

print("Model context windows (max tokens):")
for model_name, max_tokens in models:
    print(f"  {model_name}: {max_tokens:,} tokens")

print()

# Show what this means for documents
print("What can fit in GPT-4 (8,192 tokens)?")
print(f"  Short story: ~8,000 tokens (fits)")
print(f"  Novel (300 pages): ~150,000 tokens (DOES NOT fit)")
print(f"  Legal contract (50 pages): ~25,000 tokens (DOES NOT fit)")
print(f"  Year of support tickets: ~10,000,000 tokens (DOES NOT fit)")
print()