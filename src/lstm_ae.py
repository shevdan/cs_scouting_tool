
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from statistics import mean

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

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


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
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

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.mm(x.squeeze(0), self.dense_matrix)



class LSTM_AE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x
    

    def train_model(self, data_train_loader, data_val_loader, output_size, verbose, lr, epochs, clip_value, checkpoint_dir, device=None, start_epoch=0):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = MSELoss(reduction="sum")
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in range(1, epochs + 1):
            self.train()
            train_losses = []
            for sequences in tqdm(data_train_loader):
                sequences = sequences.to(device)
                sequences = sequences.squeeze()
                optimizer.zero_grad()

                # Forward pass
                outputs = self(sequences)
                loss = criterion(outputs, sequences)

                # Backward pass and optimize
                loss.backward()
                if clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            scheduler.step()
            self.eval()
            val_losses = []
            with torch.no_grad():
                for sequences in tqdm(data_val_loader):
                    sequences = sequences.to(device)
                    sequences = sequences.squeeze()
                    
                    outputs = self(sequences)
                    val_loss = criterion(outputs, sequences)
                    val_losses.append(val_loss.item())

            # Calculate mean losses
            train_loss = mean(train_losses)
            val_loss = mean(val_losses)

            if verbose:
                print(f"Epoch {epoch+start_epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if not val_loss:
                break

            # Save checkpoint if it's the best so far
            if epoch == 1 or val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch+start_epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }
                torch.save(checkpoint, f"{checkpoint_dir}/less_cols/model{output_size}_epoch_{epoch+start_epoch}_val_loss_{best_val_loss:.4f}.pt")
                print(f"Saved checkpoint to {checkpoint_dir}/less_cols/model{output_size}_epoch_{epoch+start_epoch}_val_loss_{best_val_loss:.4f}.pt")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }