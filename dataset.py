import torch
import json

class Tokenizer:
    def __init__(self, vocab, tokenizer_path=None):
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.vocab = self.special_tokens + list(vocab)
        self.vocab_size = len(self.vocab)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        
        if tokenizer_path:
            self.save(tokenizer_path)
            
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.word2idx, f)
            print("Vocabulary saved to assets")

    def encode(self, sentence):
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence.split()] 
        return tokens

    def decode(self, indices):
        return " ".join([self.idx2word.get(idx, '<UNK>') for idx in indices])

    @classmethod
    def load(cls, vocab_file):
        with open(vocab_file, 'r') as f:
            word2idx = json.load(f)

        vocab = [word for word, idx in sorted(word2idx.items(), key=lambda x: x[1])][4:] # Skip special tokens
        return cls(vocab)

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_len=128):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        english, german = self.data.iloc[idx]
        src = self.src_tokenizer.encode(english)
        tgt = self.tgt_tokenizer.encode(german)
        
        # Add <SOS> and <EOS> tokens
        src = [self.src_tokenizer.word2idx['<SOS>']] + src + [self.src_tokenizer.word2idx['<EOS>']]
        tgt = [self.tgt_tokenizer.word2idx['<SOS>']] + tgt + [self.tgt_tokenizer.word2idx['<EOS>']]
        
        # Pad the sequences
        src += [self.src_tokenizer.word2idx['<PAD>']] * (self.max_len - len(src)) 
        tgt += [self.tgt_tokenizer.word2idx['<PAD>']] * (self.max_len - len(tgt) + 1) # shift by 1
        
        return {
            'src_input_ids': torch.tensor(src[:self.max_len], dtype=torch.long),
            'tgt_input_ids': torch.tensor(tgt[:self.max_len + 1], dtype=torch.long)
        }

if __name__ == "__main__":
    # data
    import pandas as pd

    data = pd.read_csv('./data/sample_data.csv')
    corpus = " ".join((data.english + " " + data.german).tolist())
    vocab = set(corpus.split())

    # Test Tokenizer
    tokenizer = Tokenizer(vocab)
    print("Vocab size:", tokenizer.vocab_size)
    
    sentence = "Hello, how are you?"
    encoded = tokenizer.encode(sentence)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
    
    # Test Dataset
    dataset = TranslationDataset(data, tokenizer, 60)
    sample = dataset[0]
    print("Sample:", sample)
    print("Input shape:", sample['input_ids'].shape)
    print("Label shape:", sample['label'].shape)
    
    # Test DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print("Batch input shape:", batch['input_ids'].shape)
        print("Batch label shape:", batch['label'].shape)
        break