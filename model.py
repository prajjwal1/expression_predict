from torch import nn

def get_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                        d_model=config.encoder_hdim, nhead=config.encoder_nhead, dim_feedforward=config.encoder_dim_feedforward
                        ),
            num_layers=config.encoder_num_layers
    ).to(device=device)

    decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.decoder_hdim, nhead=config.decoder_nhead, dim_feedforward=config.decoder_dim_feedforward
                ),
        num_layers=config.decoder_num_layers
    ).to(device=device)
    return encoder, decoder


