import torch
import torch.nn as nn

class BiGramCharacter(object):
    """
    BiGram Character level model

    Example:
    --------
    >> import torch
    >> names = names = open('names.txt', 'r').read().splitlines()
    >> bg = BiGramCharacter(names) # Initialize
    >> bg.forward() # Create Probabilities
    >> bg.evaluate() # Evaluate using average NLL
    >> bg.generate() # Generate new names
    """
    def __init__(self, names: list):
        
        self.names = names
        # Creating string to integer mapping
        self.stoi = {y: x+1 for x,y in enumerate(sorted(set("".join(self.names))))}
        # creating entry for special character
        self.stoi["."] = 0
        # reverse mapping
        self.itos = {x: y for y, x in self.stoi.items()}

        # Initialize Probability dist matrix
        self.P = torch.zeros(27, 27, dtype = torch.int32) + 1

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
        self.P /= self.P.sum(axis = 1, keepdim=True)

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
        print(f'Log Likelihood = {log_likelihood.item():.4f}')
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
                probs = self.P[ch_idx] # Prob dist for current character
                # Sampling based on prob dist
                o = torch.multinomial(probs, 1, replacement=True, generator=self.g).item()
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
    


class DumbestNN(nn.Module):
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
        self.stoi = {y: x+1 for x,y in enumerate(sorted(set("".join(self.names))))}
        # creating entry for special character
        self.stoi["."] = 0
        # reverse mapping
        self.itos = {x: y for y, x in self.stoi.items()}

        self.g = torch.Generator().manual_seed(2024)

        self.create_training_set()
    
    def create_training_set(self):
        
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

        self.W = torch.randn((27, 27), generator = self.g, requires_grad = True)
        for k in range(epochs):
            self.xenc = nn.functional.one_hot(self.xs, num_classes=27).float()        
            logits = self.xenc @ self.W # log counts
            counts = logits.exp()
            probs = counts / counts.sum(axis = 1, keepdims= True)
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
                xgen = nn.functional.one_hot(torch.tensor([ch_idx]), num_classes=27).float()
                logits = xgen @ self.W
                counts = logits.exp()
                probs = counts / counts.sum(axis = 1, keepdims = True)
                # Sampling based on prob dist
                o = torch.multinomial(probs, 1, replacement=True, generator=self.g).item()
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
    



if __name__ == '__main__':
    
    # Names dataset
    names = open('names.txt', 'r').read().splitlines()