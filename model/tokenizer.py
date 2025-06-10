from collections import Counter
import torch
import pickle
import json

class Tokenizer:
    def __init__(self, lower=True, max_vocab_size=None, pad_token="<pad>", unk_token="<unk>", maxlen=None):
        self.lower = lower
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.maxlen = maxlen
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = None

    def fit_on_texts(self, texts):
        # Tokenize
        tokens = []
        for text in texts:
            if self.lower:
                text = text.lower()
            tokens.extend(text.split())

        # Count words and build vocab
        word_freq = Counter(tokens)
        most_common = word_freq.most_common(self.max_vocab_size)
        vocab_words = [self.pad_token, self.unk_token] + [word for word, _ in most_common]

        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            tokens = text.split()
            seq = [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences, padding="post", truncating="post"):
        pad_val = self.word2idx[self.pad_token]
        maxlen = self.maxlen or max(len(seq) for seq in sequences)

        padded = []
        for seq in sequences:
            if len(seq) < maxlen:
                pad_length = maxlen - len(seq)
                if padding == "post":
                    seq = seq + [pad_val] * pad_length
                else:
                    seq = [pad_val] * pad_length + seq
            elif len(seq) > maxlen:
                if truncating == "post":
                    seq = seq[:maxlen]
                else:
                    seq = seq[-maxlen:]
            padded.append(seq)
        return torch.tensor(padded, dtype=torch.long)

    def __len__(self):
        return len(self.word2idx)

    def vocab_size(self):
        return len(self.word2idx)
   
    def save(self, filepath):
        tokenizer_data = {
            "lower": self.lower,
            "max_vocab_size": self.max_vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "maxlen": self.maxlen,
            "word2idx": self.word2idx
        }
        with open(filepath, "w") as f:
            json.dump(tokenizer_data, f)

    @staticmethod
    def load(filepath):
        with open(filepath) as f:
            data = json.load(f)

        tokenizer = Tokenizer(
            lower=data["lower"],
            max_vocab_size=data["max_vocab_size"],
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            maxlen=data["maxlen"]
        )
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(v): k for k, v in tokenizer.word2idx.items()}
        return tokenizer