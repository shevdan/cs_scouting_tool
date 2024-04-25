
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from statistics import mean


import torch
from torch.nn import MSELoss
from tqdm import tqdm



class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        if h_activ is None:
            h_activ = nn.ReLU()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        x = x.unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()
    
class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean



class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, h_dims, h_activ=None):
        super(Decoder, self).__init__()

        if h_activ is None:
            h_activ = nn.ReLU()

        layer_dims = [latent_dim] + h_dims + [h_dims[-1]]
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.LSTM(
                input_size=layer_dims[i],
                hidden_size=layer_dims[i + 1],
                num_layers=1,
                batch_first=True
            ))
            if i < len(layer_dims) - 2:
                self.activations.append(h_activ)

        self.final_dense = nn.Linear(layer_dims[-2], output_dim)
        
        self.out_activ = nn.Tanh()  

    def forward(self, z, seq_len):
        hidden = None

        z = z.repeat(seq_len, 1).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            z, hidden = layer(z, hidden)
            if i < len(self.layers) - 1:  
                z = self.activations[i](z)

        z = self.final_dense(z)
        
        if self.out_activ:
            z = self.out_activ(z)

        return z.squeeze()


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean



class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        latent_length,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)
        self.lmbd = Lambda(hidden_size=encoding_dim,
                           latent_length=latent_length)
        self.loss_fn = MSELoss(reduction="sum")

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        latent = self.lmbd(x)
        x = self.decoder(latent, seq_len)

        return x
    
    def _rec(self, x_decoded, x, loss_fn, beta):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)
        total_loss =  recon_loss + beta * kl_loss 
        return total_loss, recon_loss, kl_loss

    def compute_loss(self, X, beta):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration

        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = X.requires_grad_(True)

        x_decoded = self(x)
        loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn, beta)

        return loss, recon_loss, kl_loss, x
    
    def train_model(
    self, data_train_loader, data_val_loader, output_size, verbose, lr, epochs, clip_value, checkpoint_dir, device=None, start_epoch=0
        ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = MSELoss(reduction="sum")
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        beta = 0.1

        for epoch in range(1, epochs + 1):
            self.train()
            train_losses, recon_losses, kl_losses = [], [], []
            for sequences in tqdm(data_train_loader):
                sequences = sequences.to(device)
                sequences = sequences.squeeze()
                optimizer.zero_grad()

                loss, recon_loss, kl_loss, _ = self.compute_loss(sequences, beta)

                loss.backward()
                if clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()

                train_losses.append(loss.item())
                recon_losses.append(recon_loss.item())
                kl_losses.append(kl_loss.item())


            # Validation phase
            scheduler.step()
            self.eval()
            val_losses, val_recon_losses, val_kl_losses = [], [], []
            with torch.no_grad():
                for sequences in tqdm(data_val_loader):
                    sequences = sequences.to(device)
                    sequences = sequences.squeeze()

                    val_loss, val_recon_loss, val_kl_loss, _ = self.compute_loss(sequences, beta)
                    val_losses.append(val_loss.item())
                    val_recon_losses.append(val_recon_loss.item())
                    val_kl_losses.append(val_kl_loss.item())
            beta = min(beta + 0.1, 1)

            # Calculate mean losses
            train_loss = mean(train_losses)
            val_loss = mean(val_losses)

            train_recon_loss = mean(recon_losses)
            val_recon_loss = mean(val_recon_losses)

            train_kl_loss = mean(kl_losses)
            val_kl_loss = mean(val_kl_losses)

            if verbose:
                print(f"Epoch {epoch+start_epoch}: Train Loss: {train_loss:.4f}, recon: {train_recon_loss:.4f}, kl: {train_kl_loss:.4f}, Val Loss: {val_loss:.4f}, recon: {val_recon_loss:.4f}, kl: {val_kl_loss:.4f}")
            if not val_loss:
                break

            if epoch == 1 or val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch+start_epoch,
                    'self_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }
                torch.save(checkpoint, f"{checkpoint_dir}/less_cols/vae/comb_loss/beta_more_dims/model{output_size}_epoch_{epoch+start_epoch}_val_loss_{best_val_loss:.4f}.pt")
                print(f"Saved checkpoint to {checkpoint_dir}/less_cols/vae/comb_loss/beta_more_dims/model{output_size}_epoch_{epoch+start_epoch}_val_loss_{best_val_loss:.4f}.pt")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }