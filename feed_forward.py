""" Feed Forward Language Model.
"""
import numpy as np
# from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as func
import torch.utils.data as dutils
from tqdm import tqdm

from utils.data import prepare_data, get_embedding


class FeedForwardModel(torch.nn.Module):
    """ Feed Forward Neural Network Language Model

    Parameters
    ----------
    n_grams : int
        Number of previous words in this language model.
    n_vocab : int
        Number of words in the whole vocabulary.
    embedding_dim : int
        The embedding dimension.
    hidden_dim : int
        The hidden dimension after the first linear layer.
    random_state : int, optional
        Integer used to assign initial random seed.

    Attributes
    ----------
    n_grams : int
        Number of previous words in this language model.
    n_vocab : int
        Number of words in the whole vocabulary.
    embedding_dim : int
        The embedding dimension.
    hidden_dim : int
        The hidden dimension after the first linear layer.
    """
    def __init__(self, n_grams, n_vocab, embedding_dim,
                 hidden_dim, random_state=None):
        super().__init__()
        if random_state:
            np.random.seed(random_state)

        self.n_grams = n_grams
        self.n_vocab = n_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding_layer = torch.nn.Embedding(n_vocab, embedding_dim)
        self.l0_linear = torch.nn.Linear(n_grams * embedding_dim, hidden_dim)
        self.l1_linear = torch.nn.Linear(hidden_dim, n_vocab)

    def forward(self, x):
        """
        """
        embed = self.embedding_layer(x)
        # reshape to (n_samples, n_grams * embedding_dim)
        cat_embed = embed.view(-1, self.n_grams * self.embedding_dim)
        out1 = func.relu(self.l0_linear(cat_embed))
        out2 = func.softmax(self.l1_linear(out1))

        return out2

    def set_embedding(self, embed_weight, requires_grad=False):
        """ Set the embedding layer using pre-trained embedding.

        Parameters
        ----------
        weight : torch tensor of shape (n_vocab, embedding_dim)
            Pre-trained weights for embedding.
        requires_grad : bool
            Indicating whether we want to update embedding.
        """
        # Verify Shape
        embed_weight = torch.nn.Parameter(
            embed_weight, requires_grad=requires_grad)
        self.embedding_layer.weight = embed_weight


def train(train_loader, model, criterion, optimizer, use_cuda=False):
    """ Train one epoch.
    """
    total_loss = 0.0

    for x_bat, y_bat in tqdm(train_loader):
        if use_cuda:
            x_bat = x_bat.cuda()
            y_bat = y_bat.cuda()
        x_var = torch.autograd.Variable(x_bat)
        y_var = torch.autograd.Variable(y_bat)

        y_pred = model(x_var)

        loss = criterion(y_pred, y_var)
        total_loss += loss.data[0] / x_bat.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss


def main():
    """ Main
    """
    use_cuda = False

    # Some limiting Parameters
    n_epochs = 10
    batch_size = 256

    n_grams = 3
    embedding_dim = 300  # This is hard coded according to the pre-trained data
    hidden_dim = 200

    # max_num_sents = 1000
    # max_num_voacb = 4000
    max_num_sents = None
    max_num_voacb = None

    vocab, x, y, x_test, y_test = prepare_data(
        n_grams, max_num_sents, max_num_voacb)
    embed_weight = get_embedding(vocab)

    # The vocabulary size uses extra space reserved for unknown word.
    n_vocab = len(vocab.keys()) + 1
    embed_weight = np.vstack([embed_weight, np.random.randn(embedding_dim)])

    # Convert to torch tensor
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    embed_weight = torch.from_numpy(embed_weight)

    model = FeedForwardModel(n_grams, n_vocab, embedding_dim, hidden_dim)
    model.set_embedding(embed_weight)
    if use_cuda:
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    train_dataset = dutils.TensorDataset(x, y)
    train_loader = dutils.DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(n_epochs):
        loss = train(train_loader, model, criterion, optimizer, use_cuda)

        print(epoch, loss)

    # y_pred = model.predict(x_test)
    # print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
