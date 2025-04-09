import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_seq_len, hidden_size)  # batch first format
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor shape (batch, seq_len, hidden)
        Returns:
            Same shape as input
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)    

'''
got to here, fix this and use for the filter output heads


'''



class MHAttentionTorch(nn.Module):
    '''
    Just a single layer MHA block using the torch implementation
    '''
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        '''
        regarding the attention masks
        attn_mask is for the query, you ONLY need this for causal masking, DO NOT use for pad masking
        key_padding_mask is for the K and is specifically for masking, use this for dealing with masking
        '''

        # Apply attention
        attn_output, _ = self.multihead_attention(
            query, key, value,
            need_weights=False,
            key_padding_mask=key_padding_mask
        )
        #layer norm
        attn_output = self.layer_norm(attn_output)
        #dropout
        attn_output = self.dropout(attn_output)
        #Add
        output = query + attn_output
        return output




class TransformerEncoderTorch(nn.Module):
    '''
    Just makes a torch transformer encoder as per attention is all you need
    '''
    def __init__(self, d_model, num_heads, num_layers, ffn_mul=4, dropout=0.1):
        super().__init__()

        if num_layers > 0:
            self.transformer_encoder = self.torch_transformer_encoder(d_model, num_heads, num_layers, ffn_mul, dropout)


    def torch_transformer_encoder(self, d_model, nhead, num_layers, ffn_mul=4, dropout=0.1):
        '''
        makes a torch transformer encoder => based on BERT, obviously with no pretrianing....
        https://pytorch.org/docs/stable/nn.html#transformer-layers
        '''
        layer = nn.TransformerEncoderLayer(d_model          = d_model, 
                                           nhead            = nhead, 
                                           batch_first      = True, 
                                           norm_first       = False, 
                                           dim_feedforward  = d_model*ffn_mul,
                                           dropout          = dropout)
        
        encoder = nn.TransformerEncoder(layer, num_layers = num_layers)
        return encoder


    def forward(self, x, mask):
        mask = mask == False
        if not hasattr(self, 'transformer_encoder'):
            return x
        else:
            try:
                return self.transformer_encoder(src=x, src_key_padding_mask=mask)
            except Exception as e:
                print(f'\ngot transformer error due to empy inputs, returning inputs as outputs')#: {e}')
                return x





class GraphTransformerModel(nn.Module):
    '''
    This makes the graph transformer using the torch transformer encoder
    It performs the concatenation and splitting of the inputs/outputs

    Not used currently
    '''
    def __init__(self, d_model, num_heads, num_layers, ffn_mul=4, dropout=0.1, skip=True):
        super().__init__()
        self.transformer = TransformerEncoderTorch(d_model, num_heads, num_layers, ffn_mul, dropout)
        self.skip = skip
        self.last_norm = nn.LayerNorm(d_model)
        self.last_dropout = nn.Dropout(dropout)
    
    def forward(self, graph_reps, graph_masks):
        # Pass through the transformer encoder
        enriched_graph_reps = self.transformer(graph_reps, graph_masks)

        #apply final dropout and layernorm
        enriched_graph_reps = self.last_norm(enriched_graph_reps)
        enriched_graph_reps = self.last_dropout(enriched_graph_reps)

        #add skip connection
        if self.skip and graph_reps.shape == enriched_graph_reps.shape:
            enriched_graph_reps = graph_reps + enriched_graph_reps  # Shape: (batch_size, top_k_spans + top_k_rels, d_model)

        return enriched_graph_reps



class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False, skip=True):
        super().__init__()

        self.lstm = nn.LSTM(input_size      = input_size,
                            hidden_size     = hidden_size,
                            num_layers      = num_layers,
                            dropout         = dropout,
                            bidirectional   = bidirectional,
                            batch_first     = True)
        self.skip = skip
        self.last_norm = nn.LayerNorm(input_size)
        self.last_dropout = nn.Dropout(dropout)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)
        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])

        #apply final dropout and layernorm
        output = self.last_norm(output)
        output = self.last_dropout(output)

        #add the skip connection
        if self.skip and output.shape == x.shape:
            output = output + x

        return output





class GraphEmbedder(nn.Module):
    """
    A module for embedding graph nodes and edges using an element-wise addition approach for distinguishing node and edge representations.
    This approach adds an identifier vector to each node and edge representation, initialized to zero, allowing the model to learn
    how best to utilize these identifiers during training. The identifiers are adjusted through training to effectively differentiate
    nodes from edges based on learned significance.

    Parameters:
    - d_model (int): The dimensionality of each input token representation, defining the size of the identifier vectors.

    Attributes:
    - node_identifier (torch.nn.Parameter): A trainable parameter that serves as the identifier for nodes, added element-wise to node representations.
    - edge_identifier (torch.nn.Parameter): A trainable parameter that serves as the identifier for edges, added element-wise to edge representations.

    Forward Inputs:
    - cand_span_reps (torch.Tensor): The tensor representing candidate span representations with shape [batch_size, num_spans, d_model].
    - cand_rel_reps (torch.Tensor): The tensor representing relation representations with shape [batch_size, num_relations, d_model].
    - cand_span_masks (torch.Tensor): The binary mask tensor for candidate spans with shape [batch_size, num_spans], indicating valid span positions.
    - rel_masks (torch.Tensor): The binary mask tensor for relations with shape [batch_size, num_relations], indicating valid relation positions.

    Output:
    - nodes (torch.Tensor): The tensor containing node representations enhanced with identifiers, with shape [batch_size, num_spans, d_model].
    - edges (torch.Tensor): The tensor containing edge representations enhanced with identifiers, with shape [batch_size, num_relations, d_model].

    Example Usage:
    module = GraphEmbedder(d_model=128)
    nodes, edges = module(cand_span_reps, cand_rel_reps, cand_span_masks, rel_masks)
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Initialize identifiers for nodes and edges to zero
        self.node_identifier = nn.Parameter(torch.zeros(d_model))
        self.edge_identifier = nn.Parameter(torch.zeros(d_model))

    def forward(self, cand_span_reps, cand_rel_reps, cand_span_masks, cand_rel_masks):
        # Apply element-wise addition of identifiers
        masked_nodes = cand_span_reps * cand_span_masks.unsqueeze(-1)
        nodes = masked_nodes + self.node_identifier

        masked_edges = cand_rel_reps * cand_rel_masks.unsqueeze(-1)
        edges = masked_edges + self.edge_identifier

        return nodes, edges
    





class SimpleProjectionLayer(nn.Module):
    '''
    Simple Projection Layer that reprojects from hidden 1 to hidden 2
    Only has dropout
    '''
    def __init__(self, input_dim, output_dim, dropout=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        #self.init_weights_alt()
        self.init_weights()

    def init_weights_alt(self):
        mean, std = 0, 0.02
        init.normal_(self.linear.weight, mean=mean, std=std)
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0)

    def init_weights(self):
        init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        return x



class FFNProjectionLayer(nn.Module):
    '''
    Projection Layer that can have nonlinearities, dropout and FFN aspects
    Always 2 linear projection layers, input and output, with a activation function between
    '''
    def __init__(self, input_dim, ffn_ratio=1.5, output_dim=None, dropout=0.2):
        super().__init__()
        #check if output dim is specified, if not then FFN outputs same dimensionality as the input
        if output_dim is None: 
            output_dim = input_dim
        intermed_dim = int(output_dim * ffn_ratio)

        self.linear_in = nn.Linear(input_dim, intermed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(intermed_dim, output_dim)
        
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.linear_in.weight, nonlinearity='relu')
        init.constant_(self.linear_in.bias, 0)
        init.kaiming_normal_(self.linear_out.weight, nonlinearity='relu')
        init.constant_(self.linear_out.bias, 0)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x




class ProjectionLayer(nn.Module):
    '''
    This is the proxy projection layer that can use either the SimpleProjectionLayer or the FFNProjectionLayer
    based on configuration.
    '''
    def __init__(self, input_dim, output_dim=None, dropout=None, layer_type='simple', ffn_ratio=None):
        super().__init__()
        # Determine which layer type to use based on the layer_type configuration
        if layer_type == 'simple':
            # Use SimpleProjectionLayer
            self.layer = SimpleProjectionLayer(input_dim  = input_dim, 
                                               output_dim = output_dim, 
                                               dropout    = dropout)
        elif layer_type == 'ffn':
            # Use FFNProjectionLayer, note that output_dim can be None which FFN handles to default to input_dim
            self.layer = FFNProjectionLayer(input_dim  = input_dim,
                                            ffn_ratio  = ffn_ratio, 
                                            output_dim = output_dim, 
                                            dropout    = dropout)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, x):
        # Delegate computation to the configured layer
        return self.layer(x)



class OutputLayerPrompt(nn.Module):
    """
    it computes interactions between item representations and type representations
    using untrainable Einstein summation.

    Parameters:
        input_size (int, optional):  The size of the input tensor last dim. Defaults to 768.
        dropout (float, optional): The dropout rate applied to the linear layer output in non-prompting mode.
                                   Defaults to 0.1.

    """
    def __init__(self, input_size=768, dropout=None):
        super().__init__()
        self.dropout = dropout


    def forward(self, item_reps, item_type_reps=None):
        if item_type_reps is None:
            raise ValueError("item_type_reps must be provided for prompting")
        return torch.einsum("bnd,btd->bnt", item_reps, item_type_reps)




class OutputLayer(nn.Module):
    """
    processes item representations through a trainable linear layer preceded by dropout.

    Parameters:
        input_size (int, optional):  The size of the input tensor last dim. Defaults to 768.
        output_size (int, optional): The size of the output last dim.  For classification => Number of types or classes for the non-prompting output layer. This must be
                                     specified if `use_prompt` is False.
                                     NOTE: will be pos and neg types for unilabel and pos types only for multilabel
        dropout (float, optional): The dropout rate applied to the linear layer output in non-prompting mode.
                                   Defaults to 0.1.
    """
    def __init__(self, input_size=768, output_size=None, dropout=None):
        super().__init__()
        self.dropout = dropout

        if output_size is None:
            raise ValueError("num_types must be provided for the output layer")
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(input_size, output_size)
        
        #self.init_weights_alt()
        self.init_weights()


    def init_weights_alt(self):
        mean, std = 0, 0.02
        init.normal_(self.output_head.weight, mean=mean, std=std)
        if self.output_head.bias is not None:
            init.constant_(self.output_head.bias, 0)

    def init_weights(self):
        init.xavier_uniform_(self.output_head.weight)
        if self.output_head.bias is not None:
            init.constant_(self.output_head.bias, 0)


    def forward(self, item_reps):
        if self.dropout is not None:
            item_reps = self.dropout(item_reps)
        return self.output_head(item_reps)




