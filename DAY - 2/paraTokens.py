import tiktoken

print("Loading tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")
print("Tokenizer loaded!\n")

print("=" * 50)
print("SECTION 4: How many tokens in real documents?")
print("=" * 50)

# A typical paragraph
paragraph = """
Artificial Intelligence is transforming the way we interact with technology. 
From chatbots to recommendation systems, AI models are becoming more sophisticated 
every day. However, these models have limitations. They cannot process unlimited 
amounts of text at once. This is why techniques like Retrieval-Augmented Generation, 
or RAG, have become essential. RAG allows AI to search through large collections 
of documents and only pull the most relevant information.
"""

tokens = tokenizer.encode(paragraph)


print(f"Paragraph length: {len(paragraph)} characters")
print(f"Number of words: {len(paragraph.split())}")
print(f"Number of tokens: {len(tokens)}")
print()