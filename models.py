'''
Transformer for Some of Power form.

Written by:
    Simo Ryu
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SOP(nn.Module):
    '''
    Sum Of Powers Transformers
    '''
    def __init__(self, chars, d_model = 512, num_layers = 6, n_vocab = 256, dim_feedforward = 1024, n_head = 8, max_len = 256, device = "cpu"):
        super().__init__()
        
        self.tok_emb_enc = nn.Embedding(n_vocab, d_model)
        self.pos_emb_enc = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward= dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers, norm = nn.LayerNorm(d_model))
        
        self.tok_emb_dec = nn.Embedding(n_vocab, d_model)
        self.pos_emb_dec = nn.Parameter(torch.zeros(1, max_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward= dim_feedforward)        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers, norm = nn.LayerNorm(d_model))

        mask = torch.triu(torch.ones(max_len, max_len), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        self.mask = mask.reshape(max_len, max_len).to(device)
        self.head = nn.Linear(d_model, n_vocab - 1)
        self.init_token = n_vocab - 1
        self.chars = chars
        self.device = device
    
    

    def forward(self, eq1, eq2):
        B, T_e = eq1.size()
        eq1 = self.tok_emb_enc(eq1)
        eq1 = self.pos_emb_enc[:, :T_e, :] + eq1
        eq1 = self.encoder(eq1.transpose(0, 1)).transpose(0, 1)

        # appended 0:
        B, T_d = eq2.size()
        eq2 = self.tok_emb_dec(eq2)
        eq2 = self.pos_emb_dec[:, :T_d, :] + eq2
        
        mask = self.mask[:T_d, :T_d]
        
        output = self.decoder(tgt = eq2.transpose(0, 1), memory = eq1.transpose(0, 1), tgt_mask = mask ).transpose(0, 1)
        output = self.head(output)
        return output
    
    @torch.no_grad()
    def toSOP(self, eq1, gen_len = 80, T = 0.02):
        chars = self.chars
        stoi = {ch : i + 1 for i, ch in enumerate(chars)}
        itos = { i + 1 : ch for i, ch in enumerate(chars)}
        itos[0] = ''

        if isinstance(eq1, str):
            x_l = [stoi[v] for v in list(eq1)]
            eq1 = torch.tensor(x_l).long().reshape(1, -1).to(self.device)
            #print(eq1)
        B, T = eq1.size()
        eq1 = self.tok_emb_enc(eq1)
        eq1 = self.pos_emb_enc[:, T, :] + eq1
        eq1 = self.encoder(eq1.transpose(0, 1)).transpose(0, 1)
        gens = torch.ones(B, 1) * self.init_token
        gens = gens.long().to(self.device)
        
        for i in range(1, gen_len + 1):
            eq2 = self.tok_emb_dec(gens)
            eq2 = self.pos_emb_dec[:, :i, :] + eq2
            out= self.decoder(tgt = eq2.transpose(0, 1), memory = eq1.transpose(0, 1), tgt_mask = self.mask[:i, :i]).transpose(0, 1)
            out = self.head(out)[:, -1, :]
            
            out = F.softmax(out/T, dim = -1)
            out = torch.multinomial(out, num_samples=1)
            out = out.reshape(B, 1)
            gens = torch.cat([gens, out], dim = 1)
        
        gens = gens.tolist()
        rets = []
        for eq in gens:
            eq = ''.join([itos[i] for i in eq[1:]])
            rets.append(eq)

        return rets
            



