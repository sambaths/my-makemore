import random

random.seed(42)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class BiGramCharacter(object):
    """
    BiGram Character level model

    Example:
    --------
    >> import torch
    >> import torch.nn as nn
    >> names = names = open('names.txt', 'r').read().splitlines()
    >> bg = BiGramCharacter(names) # Initialize
    >> bg.forward() # Create Probabilities
    >> bg.evaluate() # Evaluate using average NLL
    >> bg.generate() # Generate new names
    """

    def __init__(self, names: list):

        self.names = names
        # Creating string to integer mapping
        self.stoi = {y: x + 1 for x, y in enumerate(sorted(set("".join(self.names))))}
        # creating entry for special character
        self.stoi["."] = 0
        # reverse mapping
        self.itos = {x: y for y, x in self.stoi.items()}

        # Initialize Probability dist matrix
        self.P = torch.zeros(27, 27, dtype=torch.int32) + 1

        # Generator for reporducibility
        self.g = torch.Generator().manual_seed(2024)

    def forward(self):

        # Getting the counts based of data
        for name in self.names:
            name_ = "." + name + "."
            for ch1, ch2 in zip(name_, name_[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                self.P[ix1, ix2] += 1

        # Normalizing
        self.P = self.P * 1.0
        self.P /= self.P.sum(axis=1, keepdim=True)

    def evaluate(self):
        # Evaluation
        log_likelihood = 0.0
        n = 0
        for name in self.names:
            name_ = "." + name + "."
            for ch1, ch2 in zip(name_, name_[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                prob = self.P[ix1, ix2]
                logprob = torch.log(prob)
                log_likelihood += logprob
                n += 1
        print(f"Log Likelihood = {log_likelihood.item():.4f}")
        nll = -log_likelihood
        print(f"Negative Log Likelihood = {nll:.4f}")
        print(f"Average NLL = {nll/n:.4f}")

    def generate(self, n: int = 5):
        gen_names = []
        for _ in range(n):
            # Starting character and its index
            ch = "."
            ch_idx = self.stoi[ch]
            while True:
                probs = self.P[ch_idx]  # Prob dist for current character
                # Sampling based on prob dist
                o = torch.multinomial(
                    probs, 1, replacement=True, generator=self.g
                ).item()
                # Adding the sample to previous char
                ch += self.itos[o]
                # breaking if the sample is end of line (".")
                if o == 0:
                    break
                # Changing the current character to the sampled character
                ch_idx = o

            gen_names.append(ch[1:-1])

        for idx, name in enumerate(gen_names):
            print(f"{idx}. {name}")

        return gen_names


class DumbestNN:
    """
    In the word's of Andrej Karpathy - Dumbest, simplest Neural Net

    Example:
    --------
    >> import torch
    >> names = names = open('names.txt', 'r').read().splitlines()
    >> dumbnn = DumbestNN(names)
    >> dumbnn.forward()
    >> dumbnn.generate()

    """

    def __init__(self, names):

        self.names = names
        # Creating string to integer mapping
        self.stoi = {y: x + 1 for x, y in enumerate(sorted(set("".join(self.names))))}
        # creating entry for special character
        self.stoi["."] = 0
        # reverse mapping
        self.itos = {x: y for y, x in self.stoi.items()}

        self.g = torch.Generator().manual_seed(2024)

        self.build_dataset()

    def build_dataset(self):

        xs, ys = [], []
        for name in self.names:
            name_ = "." + name + "."
            for ch1, ch2 in zip(name_, name_[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)

        self.xs = torch.tensor(xs)
        self.ys = torch.tensor(ys)

    def forward(self, lr=1, epochs=50):

        self.W = torch.randn((27, 27), generator=self.g, requires_grad=True)
        for k in range(epochs):
            self.xenc = nn.functional.one_hot(self.xs, num_classes=27).float()
            logits = self.xenc @ self.W  # log counts
            counts = logits.exp()
            probs = counts / counts.sum(axis=1, keepdims=True)
            loss = -probs[:, self.ys].log().mean()
            print(f"Epoch {k+1} | Loss - {loss.item():.4f}")

            self.W.grad = None
            loss.backward()

            # Update
            self.W.data += -lr * self.W.grad

        self.loss = loss.item()

    def generate(self, n: int = 5):
        gen_names = []
        for _ in range(n):
            # Starting character and its index
            ch = "."
            ch_idx = self.stoi[ch]
            while True:
                xgen = nn.functional.one_hot(
                    torch.tensor([ch_idx]), num_classes=27
                ).float()
                logits = xgen @ self.W
                counts = logits.exp()
                probs = counts / counts.sum(axis=1, keepdims=True)
                # Sampling based on prob dist
                o = torch.multinomial(
                    probs, 1, replacement=True, generator=self.g
                ).item()
                # Adding the sample to previous char
                ch += self.itos[o]
                # breaking if the sample is end of line (".")
                if o == 0:
                    break
                # Changing the current character to the sampled character
                ch_idx = o

            gen_names.append(ch[1:-1])

        for idx, name in enumerate(gen_names):
            print(f"{idx}. {name}")

        return gen_names


class MLP:
    def __init__(
        self,
        names: list,
        emb_dim: int,
        n_hidden: int,
        batch_size: int,
        lr: float,
        n_steps: int,
    ):
        """
        MLP based name generator

        Example:
        --------
        >> import torch
        >> torch.nn as nn
        >> import random
        >> random.seed(42)
        >> names = open('names.txt', 'r').read().splitlines()
        >> mlp = MLP(names, emb_dim = 5, n_hidden = 100, batch_size = 32, lr = 0.1, n_steps = 20000)
        >> mlp.forward()
        >> mlp.generate()
        """

        self.names = names[:1000]
        # Creating string to integer mapping
        self.stoi = {y: x + 1 for x, y in enumerate(sorted(set("".join(self.names))))}
        # creating entry for special character
        self.stoi["."] = 0
        # reverse mapping
        self.itos = {x: y for y, x in self.stoi.items()}
        self.vocab_size = len(self.itos)
        self.batch_size = batch_size
        self.lr = lr
        self.n_steps = n_steps
        # Initialize Probability dist matrix
        self.P = torch.zeros(27, 27, dtype=torch.int32) + 1

        # Generator for reporducibility
        self.g = torch.Generator().manual_seed(2024)

        self.block_size = 3
        self.emb_dim = emb_dim
        self.n_hidden = n_hidden

        # Initialize weights
        self.C = torch.randn((self.vocab_size, self.emb_dim), generator=self.g)
        self.W1 = torch.randn(
            (self.block_size * self.emb_dim, self.n_hidden), generator=self.g
        )
        self.b1 = torch.randn(self.n_hidden, generator=self.g)
        self.W2 = torch.randn((self.n_hidden, self.vocab_size), generator=self.g)
        self.b2 = torch.randn(self.vocab_size, generator=self.g)

        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]

        for p in self.params:
            p.requires_grad = True

    def n_params(self):
        return sum([p.nelement() for p in self.params])

    def build_dataset(self, names):
        X, Y = [], []

        for name in names:
            context = [0] * self.block_size
            for ch in name + ".":
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def forward(self):

        random.shuffle(self.names)
        n1 = int(0.8 * len(self.names))
        n2 = int(0.9 * len(self.names))
        # Create train, val and test dataset
        Xtr, Ytr = self.build_dataset(self.names[:n1])
        Xval, Yval = self.build_dataset(self.names[n1:n2])
        Xte, Yte = self.build_dataset(self.names[n2:])
        lr = self.lr
        for i in range(self.n_steps):

            # minibatch
            ix = torch.randint(0, Xtr.shape[0], (self.batch_size,))

            emb = self.C[Xtr[ix]]
            o = torch.tanh(
                emb.view(-1, self.emb_dim * self.block_size) @ self.W1 + self.b1
            )
            logits = o @ self.W2 + self.b2
            loss = nn.functional.cross_entropy(logits, Ytr[ix])
            if i % 500 == 0:
                print(f"Step {i} | Loss - {loss.item():.4f}")

            for p in self.params:
                p.grad = None
            loss.backward()

            # Decay the LR after every 10000 steps by 10x, but don't decay too much
            if (i % 10000 == 0) & (lr >= 0.001):
                lr = lr / 10

            for p in self.params:
                p.data += -lr * p.grad

        # Train eval
        emb = self.C[Xtr]
        o = torch.tanh(emb.view(-1, self.emb_dim * self.block_size) @ self.W1 + self.b1)
        logits = o @ self.W2 + self.b2
        loss = nn.functional.cross_entropy(logits, Ytr)
        print(f"Train Loss - {loss:.4f}")

        # Val Eval
        emb = self.C[Xval]
        o = torch.tanh(emb.view(-1, self.emb_dim * self.block_size) @ self.W1 + self.b1)
        logits = o @ self.W2 + self.b2
        loss = nn.functional.cross_entropy(logits, Yval)
        print(f"Val Loss - {loss:.4f}")

        # Test Eval
        emb = self.C[Xte]
        o = torch.tanh(emb.view(-1, self.emb_dim * self.block_size) @ self.W1 + self.b1)
        logits = o @ self.W2 + self.b2
        loss = nn.functional.cross_entropy(logits, Yte)
        print(f"Test Loss - {loss:.4f}")

    def generate(self, n: int = 5):
        gen_names = []
        for _ in range(n):
            # Starting character and its index
            ch = ""
            context = [0] * self.block_size
            while True:
                emb = self.C[torch.tensor([context])]
                o = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = o @ self.W2 + self.b2
                probs = nn.functional.softmax(logits, dim=1)
                # Sampling based on prob dist
                ix = torch.multinomial(
                    probs, 1, replacement=True, generator=self.g
                ).item()
                context = context[1:] + [ix]
                ch += self.itos[ix]
                if ix == 0:
                    break

            gen_names.append(ch)

        for idx, name in enumerate(gen_names):
            print(f"{idx}. {name}")

        return gen_names


class Linear:
    """
    A Linear Layer similar to what we see in PyTorch
    """
    def __init__(self, n_input: int, n_output: int, bias: bool = True) -> None:
        g = torch.Generator().manual_seed(42)
        # Adding Gain also here
        self.weight = (1) * torch.randn((n_input, n_output), generator = g) / n_input**0.5
        self.bias = None
        if bias:
            self.bias = torch.zeros(n_output)

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else []) 

class BatchNorm:
    def __init__(self, n_input: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.n_input = n_input
        self.eps = eps
        self.momentum = momentum

        self.training = True

        self.gamma = torch.ones(n_input)
        self.beta = torch.zeros(n_input)

        self.running_var = torch.ones(n_input)
        self.running_mean = torch.zeros(n_input)

    def __call__(self, x):

        if self.training:
            xmean = x.mean(0, keepdim = True)
            xvar = x.var(0, keepdim = True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        self.out = self.gamma * ((x - xmean) / torch.sqrt(xvar + self.eps)) + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    

class TorchifiedMLP:
    def __init__(
        self,
        names: list,
        emb_dim: int,
        n_layers: int,
        n_hidden: int,
        batch_size: int,
        lr: float,
        n_steps: int,
    ):
        """
        Torchified MLP based name generator

        Example:
        --------
        >> import torch
        >> torch.nn as nn
        >> import random
        >> random.seed(42)
        >> names = open('names.txt', 'r').read().splitlines()
        >> mlp = TorchifiedMLP(names, emb_dim = 5, n_hidden = 100, batch_size = 32, lr = 0.1, n_steps = 20000)
        >> mlp.forward()
        >> mlp.generate()
        """

        self.names = names[:1000]
        # Creating string to integer mapping
        self.stoi = {y: x + 1 for x, y in enumerate(sorted(set("".join(self.names))))}
        # creating entry for special character
        self.stoi["."] = 0
        # reverse mapping
        self.itos = {x: y for y, x in self.stoi.items()}
        self.vocab_size = len(self.itos)
        self.batch_size = batch_size
        self.lr = lr
        self.n_steps = n_steps
        # Initialize Probability dist matrix
        self.P = torch.zeros(27, 27, dtype=torch.int32) + 1

        # Generator for reporducibility
        self.g = torch.Generator().manual_seed(42)

        self.block_size = 3
        self.emb_dim = emb_dim
        self.n_hidden = n_hidden

        # Initialize weights
        self.C = torch.randn((self.vocab_size, self.emb_dim), generator=self.g)
        self.layers = []
        # First Layer
        self.layers.append(Linear(self.block_size * self.emb_dim, n_hidden, bias = False))
        self.layers.append(BatchNorm(n_hidden))
        self.layers.append(Tanh())
        # Adding n_layers
        for _ in range(n_layers):
            self.layers.append(Linear(n_hidden, n_hidden, bias = False))
            self.layers.append(BatchNorm(n_hidden))
            self.layers.append(Tanh())
        # Last Layer / Output Layer
        self.layers.extend([Linear(n_hidden, self.vocab_size, bias = False), BatchNorm(self.vocab_size)])

        with torch.no_grad():
            self.layers[-1].gamma *= 0.1
            for layer in self.layers[:-1]:
                if isinstance(layer, Linear):
                    layer.weight *= 5/3

        self.params = [self.C] + [p for layer in self.layers for p in layer.parameters()]

        for p in self.params:
            p.requires_grad = True

    def n_params(self):
        return sum([p.nelement() for p in self.params])

    def build_dataset(self, names):
        X, Y = [], []

        for name in names:
            context = [0] * self.block_size
            for ch in name + ".":
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def forward(self):

        random.shuffle(self.names)
        n1 = int(0.8 * len(self.names))
        n2 = int(0.9 * len(self.names))
        # Create train, val and test dataset
        Xtr, Ytr = self.build_dataset(self.names[:n1])
        Xval, Yval = self.build_dataset(self.names[n1:n2])
        Xte, Yte = self.build_dataset(self.names[n2:])
        lr = self.lr
        for i in range(self.n_steps):

            # minibatch
            ix = torch.randint(0, Xtr.shape[0], (self.batch_size,), generator=self.g)

            emb = self.C[Xtr[ix]]
            x = emb.view(emb.shape[0], -1)
            for layer in self.layers:
                x = layer(x)
            loss = nn.functional.cross_entropy(x, Ytr[ix])
            if i % 500 == 0:
                print(f"Step {i} | Loss - {loss.item():.4f}")

            for layer in self.layers:
                layer.out.retain_grad()
            for p in self.params:
                p.grad = None
            loss.backward()

            # Decay the LR after every 10000 steps by 10x, but don't decay too much
            if (i % 3000 == 0) & (lr >= 0.001):
                lr = lr / 10

            for p in self.params:
                p.data += -lr * p.grad

        # Train eval
        emb = self.C[Xtr]
        x = emb.view(emb.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        loss = nn.functional.cross_entropy(x, Ytr)
        print(f"Train Loss - {loss:.4f}")

        # Val Eval
        emb = self.C[Xval]
        x = emb.view(emb.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        loss = nn.functional.cross_entropy(x, Yval)
        print(f"Val Loss - {loss:.4f}")

        # Test Eval
        emb = self.C[Xte]
        x = emb.view(emb.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        loss = nn.functional.cross_entropy(x, Yte)
        print(f"Test Loss - {loss:.4f}")

    def generate(self, n: int = 5):
        gen_names = []
        for _ in range(n):
            # Starting character and its index
            ch = ""
            context = [0] * self.block_size
            while True:
                emb = self.C[torch.tensor([context])]
                x = emb.view(emb.shape[0], -1)
                for layer in self.layers:
                    if isinstance(layer, BatchNorm):
                        layer.training = False
                probs = nn.functional.softmax(x, dim=1)

                # Sampling based on prob dist
                ix = torch.multinomial(
                    probs, 1, replacement=True, generator=self.g
                ).item()
                context = context[1:] + [ix]
                ch += self.itos[ix]
                if ix == 0:
                    break

            gen_names.append(ch)

        for idx, name in enumerate(gen_names):
            print(f"{idx}. {name}")

        return gen_names

    def activation_dist(self):

        plt.figure(figsize=(20, 4))
        legends = []
        for i, layer in enumerate(self.layers[:-1]):
            if isinstance(layer, Tanh):
                t = layer.out
                print(f"layer {str(i):5s} {layer.__class__.__name__:10s}: mean {t.mean():+.2f} std {t.std():.2f} saturation {(t.abs() > 0.97).float().mean():.2%}"  )
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} ({layer.__class__.__name__}')
            plt.legend(legends);
            plt.title('Activation Distribution')
        plt.show()
if __name__ == "__main__":

    # Names dataset
    names = open("names.txt", "r").read().splitlines()
    # mlp = TorchifiedMLP(names, emb_dim=10, n_layers=5, n_hidden=200, batch_size=64, lr=0.1, n_steps=10000)
    # mlp.forward()
    # mlp.activation_dist()
    # mlp.generate()
