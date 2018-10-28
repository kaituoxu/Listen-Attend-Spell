import torch

from models.attention import DotProductAttention

if __name__ == "__main__":
    N, To, Ti, H = 3, 5, 4, 2
    queries = torch.randn(N, To, H)
    values = torch.randn(N, Ti, H)
    attention = DotProductAttention(H)
    output, attention_distribution = attention(queries, values)
    print(output.size())
    print(output)
    print(attention_distribution.size())
    print(attention_distribution)
