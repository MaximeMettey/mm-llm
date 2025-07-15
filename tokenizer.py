class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.idx_to_token = {}
        self.vocab_size = 0
        
    def build_vocab(self, text):
        chars = sorted(set(text))
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.idx_to_token = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)
        
    def encode(self, text):
        return [self.vocab[char] for char in text]
        
    def decode(self, tokens):
        return ''.join([self.idx_to_token[token] for token in tokens])