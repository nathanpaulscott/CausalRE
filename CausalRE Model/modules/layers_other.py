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





class SimpleProjectionLayer(nn.Module):
    '''
    Simple Projection Layer that reprojects from hidden1 to hidden 2
    Only has dropout
    '''
    def __init__(self, input_dim, output_dim, dropout=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self.init_weights()

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




class TransformerEncoderTorch(nn.Module):
    '''
    Just makes a torch transformer encoder as per attention is all you need
    '''
    def __init__(self, d_model, num_heads, num_layers, ffn_mul=4, dropout=0.1):
        super(TransformerEncoderTorch, self).__init__()

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
        
        encoder = nn.TransformerEncoder(layer,
                                        num_layers = num_layers)
        return encoder


    def forward(self, x, mask):
        mask = mask == False
        if not hasattr(self, 'transformer_encoder'):
            return x
        else:
            return self.transformer_encoder(src=x, src_key_padding_mask=mask)




class GraphTransformerModel(nn.Module):
    '''
    This makes the graph transformer using the torch transformer encoder
    It performs the concatenation and splitting of the inputs/outputs

    Not used currently
    '''
    def __init__(self, d_model, num_heads, num_layers, ffn_mul=4, dropout=0.1):
        super(GraphTransformerModel, self).__init__()
        self.transformer = TransformerEncoderTorch(d_model, num_heads, num_layers, ffn_mul, dropout)

    def forward(self, node_reps, edge_reps, node_masks, edge_masks):
        # Input node_reps shape: (batch_size, top_k_spans, d_model)
        # Input edge_reps shape: (batch_size, top_k_rels, d_model)
        # Input node_masks shape: (batch_size, top_k_spans)
        # Input edge_masks shape: (batch_size, top_k_rels)

        batch_size, top_k_spans, d_model = node_reps.shape
        _, top_k_rels, _ = edge_reps.shape

        # Concatenate node and edge representations to form graph_reps
        graph_reps = torch.cat((node_reps, edge_reps), dim=1)  # Shape: (batch_size, top_k_spans + top_k_rels, d_model)

        # Combine node and edge masks
        graph_masks = torch.cat((node_masks, edge_masks), dim=1)  # Shape: (batch_size, top_k_spans + top_k_rels)

        # Pass through the transformer encoder
        enriched_graph_reps = self.transformer(graph_reps, graph_masks)

        # Split back into enriched node and edge representations
        enriched_node_reps = enriched_graph_reps[:, :top_k_spans, :]  # Shape: (batch_size, top_k_spans, d_model)
        enriched_edge_reps = enriched_graph_reps[:, top_k_spans:, :]  # Shape: (batch_size, top_k_rels, d_model)

        return enriched_node_reps, enriched_edge_reps


class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers=1, 
                 dropout=0., 
                 bidirectional=False):
        super(LstmSeq2SeqEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size      = input_size,
                            hidden_size     = hidden_size,
                            num_layers      = num_layers,
                            dropout         = dropout,
                            bidirectional   = bidirectional,
                            batch_first     = True)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)
        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
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
    




class OutputLayer(nn.Module):
    """
    A custom neural network output layer that can operate in two modes: prompting and non-prompting.
    When prompting is enabled, it computes interactions between item representations and type representations
    using untrainable Einstein summation. When prompting is disabled, it processes item representations through
    a trainable linear layer followed by dropout.

    Parameters:
        num_types (int, optional): Number of types or classes for the non-prompting output layer. This must be
                                   specified if `use_prompt` is False.
        hidden_size (int, optional): The size of the hidden layer dimensions. Defaults to 768.
        dropout (float, optional): The dropout rate applied to the linear layer output in non-prompting mode.
                                   Defaults to 0.1.
        use_prompt (bool, optional): Determines the mode of operation. If True, the layer expects to perform
                                     operations using prompting logic. If False, it uses a standard linear
                                     transformation followed by dropout. Defaults to True.

    Raises:
        ValueError: If `num_types` is not provided in non-prompting mode or if `item_type_reps` is not provided
                    in prompting mode.

    Inputs:
        item_reps (torch.Tensor): The tensor of item representations with shape (batch, num_items, hidden).
        item_type_reps (torch.Tensor, optional): The tensor of item type representations, required in prompting
                                                 mode, with shape (batch, num_types, hidden).

    Outputs:
        torch.Tensor: The output tensor. In prompting mode, the shape is (batch, num_items, num_types), representing
                      the interaction scores between items and types. In non-prompting mode, the shape is
                      (batch, num_items, num_types), where each item's representation is transformed to predict
                      its type.

    Example:
        >>> output_layer = OutputLayer(num_types=10, use_prompt=False)
        >>> logits = output_layer(item_reps)  # item_reps is a tensor of shape (batch, num_items, hidden)

        For prompting mode:
        >>> output_layer = OutputLayer(use_prompt=True)
        >>> logits = output_layer(item_reps, item_type_reps)  # item_type_reps must also be provided
    """
    def __init__(self, num_types=None, hidden_size=768, dropout=0.1, use_prompt=True):
        super(OutputLayer, self).__init__()
        self.use_prompt = use_prompt
        
        if not use_prompt:
            if num_types is None:
                raise ValueError("num_types must be provided for no prompting")
            self.output_head = nn.Linear(hidden_size, num_types)
            self.dropout = nn.Dropout(dropout)

    def forward(self, item_reps, item_type_reps=None):
        if self.use_prompt:
            if item_type_reps is None:
                raise ValueError("item_type_reps must be provided for prompting")
            return torch.einsum("bnd,btd->bnt", item_reps, item_type_reps)
        else:
            item_reps = self.dropout(item_reps)
            return self.output_head(item_reps)
