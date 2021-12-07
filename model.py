import torch
from torch import nn
import math
from typing import Optional
from torch import Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"
MAX_LENGTH = 29

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def generate_mask_for_causal_decoding(sz: int, device: str = "cpu") -> torch.Tensor:
    """
    Creates attention mask for causal decoding
    Attention mask provides indices which self attention would ignore.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask


class CustomTransformerDecoder(nn.TransformerDecoder):
    def forward(
        self,
        target_tensor: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        target_tensor_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            target_tensor (Tensor): [current_len_output x batch_size x hidden_dim]
            memory (Tensor): [len_encoded_seq x batch_size x hidden_dim]
            cache (Optional[Tensor]):
                [n_layers x (current_len_output - 1) x batch_size x hidden_dim]
        Returns:
            output (Tensor): [current_len_output x batch_size x hidden_dim]
            cache (Optional[Tensor]): [n_layers x current_len_output x batch_size x hidden_dim]
        """
        output = target_tensor

        if self.training:
            if cache is not None:
                raise ValueError("Cache should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    target_tensor_key_padding_mask=target_tensor_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            return output

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        target_tensor: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        target_tensor_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns:
            Tensor:
                During Training:
                    Returns Embedding of the whole layer: [seq_len x batch_size x hidden_dim]
                During Eval mode:
                    Returns Embedding of last token: [1 x batch_size x hidden_dim]
        """

        if self.training:
            return super().forward(
                target_tensor,
                memory,
                tgt_mask=generate_mask_for_causal_decoding(target_tensor.size(0), target_tensor.device),
                memory_mask=memory_mask,
                tgt_key_padding_mask=target_tensor_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        target_tensor_last_tok = target_tensor[-1:, :, :]

        tmp_target_tensor = self.self_attn(
            target_tensor_last_tok,
            target_tensor,
            target_tensor,
            attn_mask=None,
            key_padding_mask=target_tensor_key_padding_mask,
        )[0]
        target_tensor_last_tok = target_tensor_last_tok + self.dropout1(tmp_target_tensor)
        target_tensor_last_tok = self.norm1(target_tensor_last_tok)

        if memory is not None:
            tmp_target_tensor = self.multihead_attn(
                target_tensor_last_tok,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            target_tensor_last_tok = target_tensor_last_tok + self.dropout2(tmp_target_tensor)
            target_tensor_last_tok = self.norm2(target_tensor_last_tok)

        tmp_target_tensor = self.linear2(
            self.dropout(self.activation(self.linear1(target_tensor_last_tok)))
        )
        target_tensor_last_tok = target_tensor_last_tok + self.dropout3(tmp_target_tensor)
        target_tensor_last_tok = self.norm3(target_tensor_last_tok)
        return target_tensor_last_tok

class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()

        hidden_dim = 256
        nhead = 4
        dim_feedforward = hidden_dim * 4
        num_layers = 2
        self.vocab = vocab
        vocab_size = len(vocab)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers,
        ).to(device=DEVICE)

        self.decoder = CustomTransformerDecoder(
            CustomTransformerDecoderLayer(
                d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        ).to(device=DEVICE)

        self.classification_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, teach_forcing_tokens):
        """This function should only be used for training

        Args:
            inputs (torch.Tensor): batch_size, input_len, hidden_dim
            teach_forcing_tokens (torch.Tensor): batch_size, output_len, hidden_dim
                Each tensor needs to start with start token.
                Doesn't need to end with end token.

        Returns:
            (torch.Tensor): [description]
        """

        input_embed = self.positional_encoding(
            self.embedding(inputs).permute(1, 0, 2)
        )  # input_len, batch_size, hidden_dim

        teach_forcing_embed = self.positional_encoding(
            self.embedding(teach_forcing_tokens).permute(1, 0, 2)
        )  # output_len, batch_size, hidden_dim

        memory_mask = inputs == 0  # for source padding masks

        encoded = self.encoder(
            input_embed, src_key_padding_mask=memory_mask
        )  # input_len, batch_size, hidden_dim

        decoded = self.decoder(
            teach_forcing_embed,
            memory=encoded,
            memory_key_padding_mask=memory_mask,
        )  # output_len, batch_size, hidden_dim

        logits = self.classification_layer(decoded)  # output_len, batch_size, vocab_size
        return logits.permute(1, 0, 2)  # batch_size, output_len, vocab_size

    def predict(self, sentence_tensor):
        """
        For getting predictions (used during inference)
        """

        input_embed = self.positional_encoding(
            self.embedding(sentence_tensor).permute(1, 0, 2)
        )  # input_len, 1, hidden_dim

        memory_mask = sentence_tensor == 0

        encoded = self.encoder(
            input_embed, src_key_padding_mask=memory_mask
        )  # input_len, 1, hidden_dim

        decoded_tokens = (
            torch.LongTensor([self.vocab.idx_start_token]).to(DEVICE).unsqueeze(1)
        )  # 1, 1

        output_tokens = []
        cache = None

        while len(output_tokens) < MAX_LENGTH:  # max length of generation

            decoded_embedding = self.positional_encoding(self.embedding(decoded_tokens))

            decoded, cache = self.decoder(
                decoded_embedding,
                encoded,
                cache,
                memory_key_padding_mask=memory_mask,
            )

            logits = self.classification_layer(decoded[-1, :, :])
            new_token = logits.argmax(1).item()
            if new_token == self.vocab.idx_end_token:  # represents end of sentence
                break
            output_tokens.append(new_token)
            decoded_tokens = torch.cat(
                [
                    decoded_tokens,
                    torch.LongTensor([new_token]).unsqueeze(1).to(DEVICE),
                ],
                dim=0,
            )  # [output_len, 1]

        return decoded_tokens

