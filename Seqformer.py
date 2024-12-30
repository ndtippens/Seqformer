import math, torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset
from torch.nn.init import trunc_normal_
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(val):
    return val is not None


class AdamWarmup:
    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        
    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # update the learning rate
        self.lr = lr
        self.optimizer.step()


class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seqs = sequences
        self.lbls = labels

    def __len__(self):
        return self.lbls.shape[0]

    def __getitem__(self, idx):
        return self.seqs[idx], self.lbls[idx]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None):
        x = self.norm(x)
        context = self.norm(context) if exists(context) else x
        #qkv = self.to_qkv(x).chunk(3, dim = -1)
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, hidden_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, hidden_dim, dropout = dropout)
            ]))
    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class Seqformer(nn.Module):
    def __init__(self, *, seq_len, word_len, num_classes, dim, depth, heads, hidden_dim, channels = 4, head_dim = 64, dropout = 0.2, emb_dropout = 0.2, former=None):
        super().__init__()
        assert (seq_len % word_len) == 0

        num_words = seq_len // word_len
        patch_dim = channels * word_len
        self.dim = dim
        self.num_words = num_words
        self.word_len = word_len
        self.channels = channels

        # reshape tensor in words
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (n w) c -> b n (w c)', w = word_len),
            nn.LayerNorm(patch_dim, bias=False),
            nn.Linear(patch_dim, dim, bias=False),
            nn.LayerNorm(dim, bias=False),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_words + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        if former is None:
            self.transformer = Transformer(dim, depth, heads, head_dim, hidden_dim, dropout)
        else:
            self.transformer = former
            
        self.head = nn.Sequential(
            nn.LayerNorm(dim, bias=False),
            nn.Linear(dim, num_classes),
            #nn.Flatten()
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, 'b * d')
        return self.head(cls_tokens)


# Masked word pre-trainer
class MaskWordTask(nn.Module):
    def __init__(
        self,
        *,
        model,
        mask_p = 0.5
    ):
        super().__init__()
        assert mask_p > 0 and mask_p < 1, 'masking probability must be between 0 and 1'
        self.mask_p = mask_p

        # extract hyperparameters and functions from model to be trained
        self.model = model
        num_words, dim = model.pos_embedding.shape[-2:]

        self.to_patch = model.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*model.to_patch_embedding[1:])
        word_len = model.word_len
        channels = model.channels

        # simple linear head
        self.mask_token = nn.Parameter(torch.randn(dim))
        self.to_words = nn.Sequential(
            nn.Linear(dim, word_len*channels),
            Rearrange('b n (w c) -> b n w c', w=word_len),
            nn.Softmax(-1),
        )

    
    def forward(self, seq, mask=None):
        device = seq.device

        # get words
        words = self.to_patch(seq)
        batch, num_words, *_ = words.shape

        # for indexing purposes
        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions
        pos_emb = self.model.pos_embedding[:, 1:(num_words + 1)]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(words)
        tokens = tokens + pos_emb

        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_words)
        mask_tokens = mask_tokens + pos_emb

        # calculate which words & indices will be masked
        num_masked = int(self.mask_p * num_words)
        if exists(mask):
            masked_indices = mask
        else:
            masked_indices = torch.rand(batch, num_words, device = device).topk(k = num_masked, dim = -1).indices
        bool_mask = torch.zeros((batch, num_words), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask & encode
        tokens = torch.where(bool_mask[..., None], mask_tokens, tokens)
        encoded = self.model.transformer(tokens)
        encoded_mask_tokens = encoded[batch_range, masked_indices]

        pred = self.to_words(encoded_mask_tokens)
        labels = words[batch_range, masked_indices]
        labels = rearrange(labels, 'b n (w c) -> b n w c', w=self.model.word_len),

        return (pred, labels[0])


# masked position prediction pre-trainer
class MaskPosTask(nn.Module):
    def __init__(self, model: Seqformer, mask_p):
        super().__init__()
        self.model = model

        assert mask_p > 0 and mask_p < 1, 'masking ratio must be between 0 and 1'
        self.mask_p = mask_p

        dim = model.dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, model.num_words)
        )

    def forward(self, seq):
        device = seq.device
        tokens = self.model.to_patch_embedding(seq)
        tokens = rearrange(tokens, 'b ... d -> b (...) d')

        batch, num_words, *_ = tokens.shape

        # Masking
        num_masked = int(self.mask_p * num_words)
        rand_indices = torch.rand(batch, num_words, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens_unmasked = tokens[batch_range, unmasked_indices]

        attended_tokens = self.model.transformer(tokens, tokens_unmasked)
        logits = rearrange(self.mlp_head(attended_tokens), 'b n d -> (b n) d')
        
        # Define labels
        labels = repeat(torch.arange(num_words, device = device), 'n -> (b n)', b = batch)
        loss = F.cross_entropy(logits, labels)

        return loss