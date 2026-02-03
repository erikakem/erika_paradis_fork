import torch
from torch import nn

from model.padding import GeoCyclicPadding
from model.blocks import GMBlock


# CLP processor with structured latent
class VariationalCLP(nn.Module):
    """Convolutional layer processor with variational latent space."""

    def __init__(
        self,
        dim_in,
        dim_out,
        mesh_size,
        kernel_size=3,
        latent_dim=8,
        activation=nn.SiLU,
        expansion_factor=8,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.expansion_factor = expansion_factor

        # Encoder that produces pre latent
        self.encoder = nn.Sequential(
            # OLD CALL:
            # GMBlock(dim_in, dim_in, mesh_size),

            # Replacement call: 
            GMBlock(
                layers=['SepConv'],         # <-- one simple block
                input_dim=dim_in,
                output_dim=dim_in,              # preserve channels like before
                mesh_size=mesh_size,
                kernel_size=3,                  # same kernel for all conv-like blocks inside
                activation=False,               # only affects the *final* layer; earlier layers are auto-True
                pre_normalize=False,            # match old behavior unless you used prenorms
                ),
            nn.Conv2d(dim_in, 2 * latent_dim, kernel_size=1),  # project down

            # Replacement call: remove conv2d and instead use gmblock to keep it lightweight
            # GMBlock(
            #     layers=['FullConv'],        
            #     input_dim=dim_in,
            #     output_dim=2*latent_dim,             # <- output channel dimension can be set here
            #     mesh_size=mesh_size,
            #     kernel_size=3,                
            #     activation=False,              
            #     pre_normalize=False,            
            #     ),
        )

        # Small projection up and down in latent
        # Erika removed: make self.mu layers less expensive 
        self.mu = nn.Sequential(
            nn.Conv2d(latent_dim, self.expansion_factor * latent_dim, kernel_size=1),
            nn.LayerNorm(
                [self.expansion_factor * latent_dim, mesh_size[0], mesh_size[1]]
            ),
            activation(),
            nn.Conv2d(self.expansion_factor * latent_dim, latent_dim, kernel_size=1),
        )

        # Use GMBlock instead of LayerNorm to keep it lightweight
        # self.mu = GMBlock(
        #         layers=['FullConv'],         
        #         input_dim=dim_in,
        #         output_dim=self.expansion_factor * latent_dim,              
        #         mesh_size=mesh_size,
        #         kernel_size=3,                  
        #         activation=False,               
        #         pre_normalize=True,            
        #         )

        self.logvar = nn.Sequential(
            nn.Conv2d(latent_dim, self.expansion_factor * latent_dim, kernel_size=1),
            nn.LayerNorm(
                [self.expansion_factor * latent_dim, mesh_size[0], mesh_size[1]]
            ),
            activation(),
            nn.Conv2d(self.expansion_factor * latent_dim, latent_dim, kernel_size=1),
        )

        # Decoder that takes the concat of the logvar and mu
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, dim_in, kernel_size=1),  # project up
             GMBlock(
                layers=['SepConv'],         # <-- one simple block
                input_dim=dim_in,
                output_dim=dim_in,              # preserve channels like before
                mesh_size=mesh_size,
                kernel_size=3,                  # same kernel for all conv-like blocks inside
                activation=False,               # only affects the *final* layer; earlier layers are auto-True
                pre_normalize=False,            # match old behavior unless you used prenorms
                ),
            GeoCyclicPadding(kernel_size // 2),
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size),
        )

    def reparameterize(self, mean, log_var):
        """Reparameterization trick to sample from N(mean, var) while remaining differentiable."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, num_samples=1):
        #batch_size = x.shape[0]
       

        pre_latent = self.encoder(x)

        # Split into two groups for mu and logvar networks
        pre_mu, pre_logvar = torch.chunk(pre_latent, 2, dim=1)

        # Get distribution parameters from separate networks
        mean = self.mu(pre_mu)
        log_var = self.logvar(pre_logvar)

        # Calculate KL divergence loss against normal
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()  # Average over batch

        # Sample from latent and decode to velocity
        z = self.reparameterize(mean, log_var)

        output = self.decoder(z)

        return output, kl_loss