""" PyTorch Example for Single-Layer MLP in Ch5.2.
"""
import torch
from torch.nn import Parameter
from torch.autograd import Variable


class OneLayerMLP(torch.nn.Module):
    """ Single-Layer MLP

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.l0_weight = Parameter(torch.randn(hidden_size, n_features))
        self.l0_bias = Parameter(torch.randn(hidden_size))
        self.l1_weight = Parameter(torch.randn(1, hidden_size))
        self.l1_bias = Parameter(torch.randn(1))

    def forward(self, x):
        h = torch.tanh(torch.addmm(self.l0_bias, x, self.l0_weight.t()))
        y_pred = torch.addmm(self.l1_bias, h, self.l1_weight.t())

        return y_pred


def main():
    data = [
        ([1, 1], 1),
        ([-1, 1], -1),
        ([1, -1], -1),
        ([-1, -1], 1)
    ]
    x, y = zip(*data)
    x = Variable(torch.Tensor(x))
    y = Variable(torch.Tensor(y))

    hidden_size = 20
    n_epochs = 100

    model = OneLayerMLP(2, hidden_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        print(epoch, loss.data[0])

        # Reset the gradient to zero
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    y_pred = model(x)
    print("True Value: ", y, "Prediction Value: ", y_pred)

if __name__ == "__main__":
    main()
