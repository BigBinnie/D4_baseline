from transformers import BertModel, BertConfig
import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, vocab_size=21128):
        super().__init__()
        encoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=vocab_size,
            hidden_size=512,
            num_attention_heads=8,
        )
        self.encoder = BertModel(encoder_config)

        decoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=vocab_size,
            hidden_size=512,
            num_attention_heads=8,
        )
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.decoder = BertModel(decoder_config)

        self.linear = nn.Linear(512, vocab_size, bias=False)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        encoder_outputs = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = encoder_outputs[0]
        # out: [batch_size, max_length, hidden_size]
        outs = self.decoder(
            output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states
        )

        out = self.linear(outs[0])
        return out

    def reload_from(self, path, resize_vocab_to=None, device=None):
        state_dict = torch.load(path, map_location=device)

        if resize_vocab_to:
            extra_size = (
                resize_vocab_to
                - state_dict["encoder.embeddings.word_embeddings.weight"].shape[0]
            )
            decoder_emb = state_dict["decoder.embeddings.word_embeddings.weight"]
            device = decoder_emb.device
            encdoer_emb = state_dict["encoder.embeddings.word_embeddings.weight"]
            state_dict["decoder.embeddings.word_embeddings.weight"] = torch.cat(
                (
                    decoder_emb,
                    torch.rand(
                        extra_size,
                        decoder_emb.shape[1],
                        dtype=torch.float32,
                        device=device,
                    ),
                ),
                dim=0,
            )
            extra_size = resize_vocab_to - encdoer_emb.shape[0]
            state_dict["encoder.embeddings.word_embeddings.weight"] = torch.cat(
                (
                    encdoer_emb,
                    torch.rand(
                        extra_size,
                        encdoer_emb.shape[1],
                        dtype=torch.float32,
                        device=device,
                    ),
                ),
                dim=0,
            )
            extra_size = resize_vocab_to - state_dict["linear.weight"].shape[0]
            linear_layer = state_dict["linear.weight"]
            state_dict["linear.weight"] = torch.cat(
                (
                    linear_layer,
                    torch.rand(extra_size, 512, dtype=torch.float32, device=device),
                ),
                0,
            )
        self.load_state_dict(state_dict, strict=False)
