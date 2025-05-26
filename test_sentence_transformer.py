# test_sentence_transformer.py
try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformers imported successfully.")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded successfully!")
    print("If you see this, PyTorch and SentenceTransformers are likely working.")
except Exception as e:
    print(f"Error: {e}")