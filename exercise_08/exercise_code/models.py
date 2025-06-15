import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        # Initialize your encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hparams["n_hidden"]),
            nn.BatchNorm1d(hparams["n_hidden"]),
            nn.ReLU(),
            nn.Linear(hparams["n_hidden"], hparams["n_hidden"]),
            nn.BatchNorm1d(hparams["n_hidden"]),
            nn.ReLU(),
            nn.Linear(hparams["n_hidden"], hparams["latent_dim"])   # The latent_dim should be the output size of your encoder
        ) 
        

        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        # Initialize your decoder  
        self.decoder = nn.Sequential(
            nn.Linear(hparams["latent_dim"], hparams["n_hidden"]),
            nn.BatchNorm1d(hparams["n_hidden"]),
            nn.ReLU(),
            nn.Linear(hparams["n_hidden"], hparams["n_hidden"]),
            nn.BatchNorm1d(hparams["n_hidden"]),
            nn.ReLU(),
            nn.Linear(hparams["n_hidden"], output_size),   # The latent_dim should be the output size of your encoder
        )


    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):                
        # Feed the input image to encoder to generate the latent vector. 
        # Then decode the latent vector and get your reconstructionof the input.                                                       #
        latent_vector = self.encoder(x)
        reconstruction = self.decoder(latent_vector)

        return reconstruction

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(
            self.parameters(recurse=True),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        # Set the model to training mode
        self.train()
        images = batch
        images = images.to(self.hparams["device"])
        # Reshape the input to fit fully connected layers
        images = images.view(images.shape[0], -1)

        # Reset the gradients before each training step
        self.optimizer.zero_grad()
        pred = self.forward(images)
        loss = loss_func(pred, images)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        # Set the model to training mode
        self.eval()
        with torch.no_grad():
            images = batch
            images = images.to(self.hparams["device"])
            # Reshape the input to fit fully connected layers
            images = images.view(images.shape[0], -1)
            pred = self.forward(images)
            loss = loss_func(pred, images)
            # Reset the gradients before each training step
            self.optimizer.zero_grad()

        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Adding classifier layers after the encoder
        self.model = nn.Sequential(
            nn.Linear(hparams["latent_dim"], hparams["num_classes"]),  
        )
  
        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(
            self.parameters(recurse=True),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )


    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc

