import torch
from torch import nn
from .layers import MLP



class RelationRep(nn.Module):
    '''
    So this is just taking in the candidate span_reps, 
    halving the hidden dim, then concatenating the head and tail span rep for every possible combination to get teh span reps
    There is no thought to including context, i.e. reps for tokens between the head and tail spans, or within a window before after each head and tail!!!!!
    '''
    def __init__(self, hidden_size, dropout, ffn_mul):
        super().__init__()

        self.head_mlp = nn.Linear(hidden_size, hidden_size // 2)
        self.tail_mlp = nn.Linear(hidden_size, hidden_size // 2)
        self.out_mlp = MLP([hidden_size, hidden_size * ffn_mul, hidden_size], dropout)

    def forward(self, span_reps):
        """
        input span_reps [B, topk, D]
        return relation_reps [B, topk, topk, D]
        """

        heads, tails = span_reps, span_reps

        # Apply MLPs to heads and tails
        #Nathan - reduce their hidden dims by half for concatenation
        heads = self.head_mlp(heads)
        tails = self.tail_mlp(tails)

        # Expand heads and tails to create relation representations
        '''
        #this is copying the middle dim in a different way for heads and tails so as to facilitate all possible combos later, the dims of heads/tails will now be (B, max_top_k, max_top_k, H//2)
        #instead should do:
        heads, tails = heads.unsqueeze(2), tails.unsqueeze(1)
        relation_reps = torch.cat([heads, tails], dim=-1)
        '''
        heads = heads.unsqueeze(2).expand(-1, -1, heads.shape[1], -1)
        tails = tails.unsqueeze(1).expand(-1, tails.shape[1], -1, -1)
        # Concatenate heads and tails to create relation representations
        #concat the head and tail for every combination
        relation_reps = torch.cat([heads, tails], dim=-1)

        # Apply MLP to relation representations
        #just a FFN to be clear
        relation_reps = self.out_mlp(relation_reps)

        return relation_reps



class RelationRep(nn.Module):
    def __init__(self, hidden_size, dropout, ffn_mul):
        super().__init__()

        #layers to halve hidden dim of head and tail for concatenation
        self.head_mlp = nn.Linear(hidden_size, hidden_size // 2)
        self.tail_mlp = nn.Linear(hidden_size, hidden_size // 2)
        #this is just a FFN like in transformers, should have named it that!!!!
        self.out_mlp = MLP([hidden_size, hidden_size * ffn_mul, hidden_size], dropout)

    def forward(self, span_reps):
        """
        :param span_reps [B, topk, D]
        :return relation_reps [B, topk, topk, D]
        """

        heads, tails = span_reps, span_reps

        # Apply MLPs to heads and tails
        heads = self.head_mlp(heads)
        tails = self.tail_mlp(tails)

        # Expand heads and tails to create relation representations
        heads = heads.unsqueeze(2).expand(-1, -1, heads.shape[1], -1)
        tails = tails.unsqueeze(1).expand(-1, tails.shape[1], -1, -1)

        # Concatenate heads and tails to create relation representations
        relation_reps = torch.cat([heads, tails], dim=-1)

        # Apply MLP to relation representations
        relation_reps = self.out_mlp(relation_reps)

        return relation_reps
