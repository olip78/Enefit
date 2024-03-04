from torch.nn import functional as F
from torch import nn
import torch

momentum = 0.1

# ------- self attention --------

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, rel_emb, device):
        super(SelfAttention, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (modules.__contains__(module)), "Invalid module"

        if module == 'spatial' or module == 'temporal':
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size], device=self.device))

        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim], device=self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size, device=self.device)

    def forward(self, v, k, q):
        N, _, _ = v.shape

        #non-shared weights between heads for spatial and temporal modules
        if self.module == 'spatial' or self.module == 'temporal':
            values = self.values(v)
            keys = self.keys(k)
            queries = self.queries(q)
            values = values.reshape(N, self.seq_len, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len, self.heads, self.embed_size)

        #shared weights between heads for spatio-temporal module
        else:
            values, keys, queries = v, k, q
            values = values.reshape(N, self.seq_len, self.heads, self.head_dim)
            keys = keys.reshape(N, self.seq_len, self.heads, self.head_dim)
            queries = queries.reshape(N, self.seq_len, self.heads, self.head_dim)
            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

        if self.rel_emb:
            QE = torch.matmul(queries.transpose(1, 2), self.E.transpose(1,2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len, self.seq_len)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device),
                    1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S

        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device),
                    1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3)

        if self.module == 'spatial' or self.module == 'temporal':
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len*self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len, self.heads*self.head_dim)

        z = self.fc_out(z)

        return z

    def _mask_positions(self, qe):
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=self.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]


class CrossAttention(nn.Module):
    """encoder -> memory dim: N x d (seq_len_vk)
       decoder dim: M x d (seq_len_q)
    """
    def __init__(self, embed_size, heads, 
                 seq_len_vk,
                 seq_len_q, 
                 module, rel_emb, device):
        super(CrossAttention, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len_vk = seq_len_vk
        self.seq_len_q = seq_len_q
        self.module = module
        self.rel_emb = rel_emb
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (modules.__contains__(module)), "Invalid module"

        if module == 'spatial' or module == 'temporal':
            self.head_dim = self.seq_len_q
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size], device=self.device))

        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len_vk, self.head_dim], device=self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size, device=self.device)

    def forward(self, v, k, q):
        N, _, _ = v.shape

        #non-shared weights between heads for spatial and temporal modules
        if self.module == 'spatial' or self.module == 'temporal':
            values = self.values(v)
            keys = self.keys(k)
            queries = self.queries(q)
            values = values.reshape(N, self.seq_len_vk, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len_vk, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len_q, self.heads, self.embed_size)

        #shared weights between heads for spatio-temporal module
        else:
            values, keys, queries = v, k, q
            values = values.reshape(N, self.seq_len_vk, self.heads, self.head_dim)
            keys = keys.reshape(N, self.seq_len_vk, self.heads, self.head_dim)
            queries = queries.reshape(N, self.seq_len_q, self.heads, self.head_dim)
            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

        if self.rel_emb:
            QE = torch.matmul(values.transpose(1, 2), self.E.transpose(1,2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len_q, self.seq_len_vk)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len_q, self.seq_len_vk, device=self.device),
                    1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S
        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3)

        if self.module == 'spatial' or self.module == 'temporal':
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len_q*self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len_q, self.heads*self.head_dim)

        z = self.fc_out(z)

        return z

    def _mask_positions(self, qe):
        L1, L2 = qe.shape[-2], qe.shape[-1]
        mask = torch.triu(torch.ones(L1, L2, device=self.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]

# ------ Encoder -----------

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb, device):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb, device=device)

        if module == 'spatial' or module == 'temporal':
            self.norm1 = nn.BatchNorm1d(seq_len*heads, momentum=momentum, track_running_stats=False)
            self.norm2 = nn.BatchNorm1d(seq_len*heads, momentum=momentum, track_running_stats=False)
            #self.norm1 = nn.LayerNorm(embed_size)
            #self.norm2 = nn.LayerNorm(embed_size)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len, track_running_stats=False)
            self.norm2 = nn.BatchNorm1d(seq_len, track_running_stats=False)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

    def forward(self, x):
        attention = self.attention(x, x, x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class EncoderModule(nn.Module):
    def __init__(self, seq_len, embed_size, num_layers, heads, device,
                 forward_expansion, module,
                 rel_emb=True):
        super(EncoderModule, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.layers = nn.ModuleList(
            [
             EncoderBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, 
                          rel_emb = rel_emb, device=device)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
        out = self.fc_out(out)
        return out


class Encoder(nn.Module):
    def __init__(self,
                 input_shape,
                 output_size,
                 embed_size,
                 num_layers, forward_expansion, heads, dropout, device):
        super(Encoder, self).__init__()

        _, self.seq_len, self.num_var, self.data_size = input_shape
        self.num_elements = self.seq_len*self.num_var
        self.embed_size = embed_size
        self.output_size = output_size
        self.device = device

        self.element_embedding = nn.Linear(self.data_size*self.seq_len, embed_size*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)

        self.temporal = EncoderModule(seq_len=self.seq_len,
                                embed_size=embed_size,
                                num_layers=num_layers,
                                heads=self.num_var,
                                device=device,
                                forward_expansion=forward_expansion,
                                module='temporal',
                                rel_emb=True)

        self.spatial = EncoderModule(seq_len=self.num_var,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=self.seq_len,
                               device=device,
                               forward_expansion=forward_expansion,
                               module = 'spatial',
                               rel_emb=True)

        self.spatiotemporal = EncoderModule(seq_len=self.seq_len*self.num_var,
                                      embed_size=embed_size,
                                      num_layers=num_layers,
                                      heads=heads,
                                      device=device,
                                      forward_expansion=forward_expansion,
                                      module = 'spatiotemporal',
                                      rel_emb=True)

        self.temporal_dropout = nn.Dropout(p=dropout)
        self.spatial_dropout = nn.Dropout(p=dropout)
        self.spatiotemporal_dropout = nn.Dropout(p=dropout)

        # consolidate embedding dimension
        self.n_compressor = 2 * self.num_var*self.embed_size
        self.compressor = nn.Sequential(
            nn.Linear(self.n_compressor, self.output_size[1]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.seq_len, momentum=momentum, track_running_stats=False),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        #process/embed input for temporal module
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.temporal_dropout(self.pos_embedding(positions) + x_temporal)
        #x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, dropout)

        #process/embed input for spatial module
        x_spatial = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.spatial_dropout(self.variable_embedding(vars) + x_spatial)
        #x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, dropout)

        #process/embed input for spatio-temporal module
        #x_spatio_temporal = self.element_embedding(x).reshape(batch_size, self.seq_len*self.num_var, self.embed_size)
        #positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var* self.seq_len).to(self.device)
        #x_spatio_temporal = self.spatiotemporal_dropout (self.pos_embedding(positions) + x_spatio_temporal)

        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        out = torch.cat((out1, out2), dim=-1)
        out = out.reshape(batch_size, self.output_size[0], -1)
        out = self.compressor(out)

        return out


# ------ Decoder ----------

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len_vk, seq_len_q, module, rel_emb, device):
        super(CrossAttentionBlock, self).__init__()
        self.attention2 = CrossAttention(embed_size, heads, seq_len_vk, seq_len_q, module, 
                rel_emb=rel_emb, device=device)
        self.norm2 = nn.BatchNorm1d(seq_len_q*heads, momentum=momentum, track_running_stats=False)
        self.norm3 = nn.BatchNorm1d(seq_len_q*heads, momentum=momentum, track_running_stats=False)
        #self.norm2 = nn.LayerNorm(embed_size)
        #self.norm3 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, x, e):
        attention = self.attention2(e, e, x)
        x = self.norm2(attention + x)
        out = self.feed_forward(x)
        out = out + x
        out = self.norm3(out)
        return out

class CrossAttentionModule(nn.Module):
    def __init__(self, seq_len_vk, seq_len_q, embed_size, num_layers, heads, module, device,
                 rel_emb=True):
        super(CrossAttentionModule, self).__init__()
        self.embed_size = embed_size
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.layers = nn.ModuleList(
            [
             CrossAttentionBlock(embed_size, heads, seq_len_vk, seq_len_q, module,
                          rel_emb = rel_emb, device=device)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x, e):
        for layer in self.layers:
            out = layer(x, e)
        out = self.fc_out(out)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 input_shape,
                 embed_size,
                 e_memory_len,
                 output_size,
                 heads,
                 num_layers, forward_expansion, dropout, device):
        super(Decoder, self).__init__()

        _, self.seq_len, self.num_var, self.data_size = input_shape
        self.num_elements = self.seq_len*self.num_var
        self.embed_size = embed_size
        self.output_size = output_size
        self.device = device

        self.element_embedding = nn.Linear(self.data_size*self.seq_len, embed_size*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)

        num_layers_e, num_layers_ca = num_layers

        self.temporal = EncoderModule(seq_len=self.seq_len,
                                embed_size=embed_size,
                                num_layers=num_layers_e,
                                heads=self.num_var,
                                device=device,
                                forward_expansion=forward_expansion,
                                module='temporal',
                                rel_emb=True)

        self.spatial = EncoderModule(seq_len=self.num_var,
                               embed_size=embed_size,
                               num_layers=num_layers_e,
                               heads=self.seq_len,
                               device=device,
                               forward_expansion=forward_expansion,
                               module = 'spatial',
                               rel_emb=True)

        self.temporal_dropout = nn.Dropout(p=dropout)
        self.spatial_dropout = nn.Dropout(p=dropout)

        self.n_compressor = 2 * self.num_var*self.embed_size
        self.compressor = nn.Sequential(
            nn.Linear(self.n_compressor, self.embed_size),
            nn.LeakyReLU(),
            #nn.LayerNorm(self.embed_size),
            nn.BatchNorm1d(self.seq_len, momentum=momentum, track_running_stats=False),
            nn.Dropout(p=dropout)
        )

        self.decoder = CrossAttentionModule(seq_len_vk=e_memory_len,
                                seq_len_q=self.seq_len,
                                embed_size=self.embed_size,
                                num_layers=num_layers_ca,
                                heads=heads,
                                module = 'temporal',
                                device=device,
                                rel_emb=True)

        self.fc_out = nn.Sequential(
                                    nn.Linear(self.embed_size, self.embed_size // 2),
                                    nn.LeakyReLU()
                                    )

        #prediction
        n = 24 * (self.embed_size // 2)
        self.regressor = nn.Sequential(
                                     nn.Flatten(),
                                     nn.Linear(n, self.output_size),
                                    )


    def forward(self, x, embedding):
        batch_size = x.shape[0]

        #process/embed input for temporal module
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.temporal_dropout(self.pos_embedding(positions) + x_temporal)
        #x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, dropout)

        #process/embed input for spatial module
        x_spatial = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.spatial_dropout(self.variable_embedding(vars) + x_spatial)
        #x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, dropout)

        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        
        out = torch.cat((out1, out2), dim=-1)
        out = out.reshape(batch_size, 24, -1)
        out = self.compressor(out)

        out = self.decoder(out, embedding)
        out = self.fc_out(out)
        out = self.regressor(out)

        return out


class Transformer(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Transformer, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
    
    def forward(self, X, H):
        X = X.reshape(X.shape[0], -1, X.shape[2]).transpose(2, 1)
        H = H.reshape(H.shape[0], -1, H.shape[2]).transpose(2, 1)
        
        embedding = self.encoder(X)
        output = self.decoder(H, embedding)
        return output


class weighted_MSELoss(nn.Module):
    def __init__(self, weights, device):
        super().__init__()
        self._weights = weights
        self._weights[0] = torch.tensor(self._weights[0], dtype=torch.float32).to(device)
        self._weights[1] = torch.tensor(self._weights[1], dtype=torch.float32).to(device)
    def forward(self, output, target, units):
        weights = torch.cat([self._weights[1][units].unsqueeze(1).expand(-1, 24), 
                             self._weights[0][units].unsqueeze(1).expand(-1, 24)], dim=-1)
        loss = ((output - target)**2 ) * weights
        return loss.mean()
