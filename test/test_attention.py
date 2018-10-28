import torch

from models.attention import DotProductAttention

if __name__ == "__main__":
    torch.manual_seed(123)
    Tos = [1, 5]
    for i in range(len(Tos)):
        print("\n### loop", i)
        N, To, Ti, H = 3, Tos[i], 4, 2
        queries = torch.randn(N, To, H)
        values = torch.randn(N, Ti, H)
        attention = DotProductAttention()
        attention_output, attention_distribution = attention(queries, values)
        print(attention_output.size())
        print(attention_output)
        print(attention_distribution.size())
        print(attention_distribution)
