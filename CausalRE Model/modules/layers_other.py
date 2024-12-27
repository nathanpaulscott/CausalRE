import torch
from torch import nn
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

        you could try using this for various things:
        - the attention pooling => just use less layers (mayb just 1) and less heads (say 4) => I say no, the random query vector with mha and ffn is simpler and better
        - the main encoder => I think no need as this is not pretrained and the HF or flair options are more optimised and allow usage of different architectures
        - graph attention?  Maybe
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
    def __init__(self, d_model):
        super().__init__()

        # Project node to half of its dimension
        #this is used to form the node specific identifier
        #the weights are trainable, so I guess that is enough!!
        #He doesn't init the weights
        self.project_node = nn.Linear(d_model, d_model // 2)

        # Initialize identifier with zeros
        #this is the node/edge discriminator identifier
        #nn.Parameter() this just makes this 2D tensor trainable
        self.identifier = nn.Parameter(torch.randn(2, d_model))
        nn.init.zeros_(self.identifier)

    def forward(self, candidate_span_rep):
        """
        This forms the node and edge reps from the candidate span reps
        
        The inputs are the candidate span reps
        
        The outputs are the node and edge reps, which are similar to the pre-graph span and relation reps, but they have the graph identifiers added to them which gives graph position

        NAthan:
        This code has an error (read point 5)
        There is a whole bunch or weird shit going on here:
        (1) he is doing the same relation generation algo he just did prior to building the graph, so why fucking do it again in here?  Just pass the rel reps in!!!
        (2) he doesn't mask out self relations/edges, why not?  
        (3) the nodes is stupid, he projects span reps down to hidden/2 then concatenates them to each other, that is just dumb, why not just use the plain span reps for nodes?  Makes no sense
        (4) the identifiers, he uses one id rep for all nodes and another id rep for all edges, so if the identifier reps are just for differentiating nodes from edges, then this is a dumb way of doing it
        Just add an identifier dim to the rep, just cat 0 for nodes and 1 for edges.
        (5) in the paper he states that he adds the raw span reps to the node identifier reps but he doesn't do that
        he actually should add the raw rel reps to the edges also, he even mentions that it works better if he does that in the paper
        
        So basically this graph embedder is not finished.
        """
        max_top_k = candidate_span_rep.size()[1]

        # Project nodes
        #project span reps to hidden/2, a lower dimensionality
        #this is actually for the node specific identifier, he is just init the node specific identifiers with the projected span_reps
        #I would have said it makes more sense to use actual orthogonal codes, but hey
        nodes = self.project_node(candidate_span_rep)

        # Split nodes into heads and tails
        #Nathan: do the same thing we did for the regular span-pair expansion, add a dim and put a copy of the span reps in one or the other
        heads = nodes.unsqueeze(2).expand(-1, -1, max_top_k, -1)
        tails = nodes.unsqueeze(1).expand(-1, max_top_k, -1, -1)
        # Concatenate heads and tails to form edges
        # then concate the heads and tails tensors to make the relations (edges), 
        #edges will be the edge identifiers (which are just concat of the head and tail node identifier)
        edges = torch.cat([heads, tails], dim=-1)

        # Duplicate nodes along the last dimension
        #Nathan: These are basically the self edges, not sure the reason for making them
        #NOTE that these will also be in the edges tensor on the diagonals of the middle 2 dims, so I do not know why he makes it again!?
        #anyway, these are the node specific identifiers
        nodes = torch.cat([nodes, nodes], dim=-1)

        #Add node/edge identifier to nodes and edges
        #here he adds the node/edge discriminator identifier so we can tell an edge from a node
        nodes += self.identifier[0]
        edges += self.identifier[1]

        #why is he not adding the raw span/rel reps to this node/edge identifier reps?
        #that is what they do in the paper!!!!
        #HE SHOUD HAVE THIS HERE, but he doesn't:
        #HE SHOUD HAVE THIS HERE, but he doesn't:
        #HE SHOUD HAVE THIS HERE, but he doesn't:
        #HE SHOUD HAVE THIS HERE, but he doesn't:
        #nodes += candidate_span_rep
        
        return nodes, edges

