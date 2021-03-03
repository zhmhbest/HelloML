from attention.common_header import *
from torch.autograd import Variable


class Embeddings(nn.Module):
    """
        嵌入
        - 《Using the Output Embedding to Improve Language Models》 https://arxiv.org/abs/1608.05859
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
        位置信息
        - 《Convolutional Sequence to Sequence Learning》 https://arxiv.org/abs/1705.03122
    """

    def __init__(
            self,
            d_model: int,
            dropout: float = 0,
            max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        # shape = [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)
        # = [[0], [1], ..., [max_len-1]]

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model)
        )
        # len(div_term) = d_model // 2

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # shape = [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        x = x + Variable(self.pe[:, :x.shape[1]], requires_grad=False)
        return self.dropout(x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    _pe = PositionalEncoding(20)

    _y = _pe.forward(Variable(torch.zeros(1, 100, 20)))
    print(_y.shape)

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(100), _y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()
