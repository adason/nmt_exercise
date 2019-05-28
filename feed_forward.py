""" Feed Forward Language Model.
"""
import numpy as np
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
        """ Forward computation.
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


def train(train_loader, model, criterion, optimizer, use_cuda):
    """ Train one epoch.
    """
    total_loss = 0.0
    n_sample = 0

    for x_bat, y_bat in tqdm(train_loader):
        if use_cuda:
            x_bat = x_bat.cuda()
            y_bat = y_bat.cuda()
        x_bat = torch.autograd.Variable(x_bat)
        y_bat = torch.autograd.Variable(y_bat)

        y_pred_bat = model(x_bat)

        loss = criterion(y_pred_bat, y_bat)
        total_loss += loss.data.item()
        n_sample += x_bat.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = total_loss / n_sample

    return loss


def validate(val_loader, model, criterion, use_cuda):
    """ Validate Test Data
    """
    total_loss = 0.0
    n_correct = 0
    n_sample = 0

    for x_bat, y_bat in tqdm(val_loader):
        if use_cuda:
            x_bat = x_bat.cuda()
            y_bat = y_bat.cuda()
        x_bat = torch.autograd.Variable(x_bat)
        y_bat = torch.autograd.Variable(y_bat)

        y_pred_bat = model(x_bat)

        loss = criterion(y_pred_bat, y_bat)
        total_loss += loss.data.item()
        n_sample += x_bat.size(0)

        _, predicted = torch.max(y_pred_bat.data, 1)
        n_correct += (predicted == y_bat.data).sum()

    accuracy = 100.0 * n_correct / n_sample
    loss = total_loss / n_sample

    return loss, accuracy


def main():
    """ Main
    """
    use_cuda = torch.cuda.is_available()

    # Some limiting Parameters
    n_epochs = 10
    batch_size = 256

    n_grams = 5
    embedding_dim = 300  # This is hard coded according to the pre-trained data
    hidden_dim = 200

    # max_num_sents = 1000
    # max_num_voacb = 4000
    max_num_sents = None
    max_num_voacb = None

    vocab, x, y, x_test, y_test = prepare_data(
        n_grams, max_num_sents, max_num_voacb)
    embed_weight = get_embedding(vocab)
    # embed_weight = np.random.randn(len(vocab.keys()), embedding_dim)  # random embedding

    # The vocabulary size uses extra space reserved for unknown word.
    n_vocab = len(vocab.keys()) + 1
    embed_weight = np.vstack([embed_weight, np.random.randn(embedding_dim)])

    # Convert to torch tensor
    # Note that pytorch preferred to use float32 for numeric values and int64(LongIng) for index
    # So for the pre-trained embedding weight, we will need to explicitly convert into float
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    embed_weight = torch.from_numpy(embed_weight).float()

    model = FeedForwardModel(n_grams, n_vocab, embedding_dim, hidden_dim)
    model.set_embedding(embed_weight)
    if use_cuda:
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss(size_average=False)
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    train_dataset = dutils.TensorDataset(x, y)
    train_loader = dutils.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = dutils.TensorDataset(x_test, y_test)
    test_loader = dutils.DataLoader(test_dataset, batch_size)

    for epoch in range(n_epochs):
        print("Training Epoch: ", epoch)

        loss = train(train_loader, model, criterion, optimizer, use_cuda)
        print("Train Loss: ", loss)

        test_loss, accuracy = validate(test_loader, model, criterion, use_cuda)
        print("Test Loss: ", test_loss, "Test Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
