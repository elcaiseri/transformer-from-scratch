import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, 'd_model should be divisible by num_heads'
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
    
    def attention_score(self, q, k, v, mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        
        return out
        
    def forward(self, q, k, v, mask=None):
        B, S, D = q.size()
        
        Q = self.W_q(q).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        
        out = self.attention_score(Q, K, V, mask)
        out = out.transpose(1, 2).reshape(B, S, D)
        #out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        return self.W_o(out)
        
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        x1 = self.mha(x, x, x, mask)
        x2 = self.norm1(x + self.dropout(x1)) 
        x3 = self.ff(x2)
        out = self.norm2(x2 + self.dropout(x3))
        
        return out
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        B, S, D = x.size()
            
        x1 = self.mha1(x, x, x, tgt_mask)
        x2 = self.norm1(x + self.dropout(x1))
        
        x3 = self.mha2(x2, enc_out, enc_out, src_mask)
        x4 = self.norm2(x2 + self.dropout(x3))
        
        x5 = self.ff(x4)
        
        out = self.norm3(x4 + self.dropout(x5))
        return out

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, src_enc, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_enc, tgt_mask, src_mask)
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers, d_model, num_heads, d_ff, dropout):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embedding = nn.Embedding(d_model, d_model)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src, tgt: (B, S)
        B, S = src.size()
        pos = torch.arange(S, device=src.device).unsqueeze(0).repeat(B, 1) # (B, S)

        src = self.src_embedding(src) + self.pos_embedding(pos) # (B, S, D) + (B, S, D) -> (B, S, D)
        tgt = self.tgt_embedding(tgt) + self.pos_embedding(pos)
        
        src_enc = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, src_enc, tgt_mask, src_mask)
        
        out = self.linear(dec_out)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    seq_len = 10
    vocab_size = 100
    
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=src.device)).unsqueeze(0)
    
    transformer = Transformer(vocab_size, vocab_size, num_layers, d_model, num_heads, d_ff, dropout).to(device)
    out = transformer(src, tgt, mask, mask)
    print("transformer out shape:", out.shape)
