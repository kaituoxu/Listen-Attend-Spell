import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    r"""Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.

    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self, dim):
        super(DotProductAttention, self).__init__()
        # TODO: move this out of this class?
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H

        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(
            attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        attention_output = torch.bmm(attention_distribution, values)
        # concat -> (N, To, 2*H)
        concated = torch.cat((attention_output, queries), dim=2)
        # TODO: Move this out of this class?
        # output -> (N, To, H)
        output = torch.tanh(self.linear_out(
            concated.view(-1, 2*hidden_size))).view(batch_size, -1, hidden_size)

        return output, attention_distribution
