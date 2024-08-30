import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layer=1, drop_out=0.0, bidirectional=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layer = num_hidden_layer
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_hidden_layer, batch_first=True,
                           dropout=drop_out, bidirectional=bidirectional)

    def forward(self, x):
      output, (hidden, cell) = self.rnn(x)
      return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.rnn_cell = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        # input shape: (batch_size,)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)

        # Add time dimension
        # input shape: (batch_size, 1, output_dim)
        input = input.unsqueeze(1)

        # output shape: (batch_size, 1, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        output, hidden = self.rnn_cell(input, hidden)

        # attn_input shape: (batch_size, 1, hidden_dim)
        # attn_input = hidden[-1].unsqueeze(1)

        # Calculate attention weights
        # alighment_scores shape: (batch_size, 1, seq_len)
        alighment_scores = torch.bmm(self.attn(output), encoder_outputs.transpose(1, 2)) # encoder_outputs.transpose(1,2) shape: (batch_size, hidden_dim, seq_len)
        # attn_weights shape: (batch_size, 1, seq_len)
        attn_weights = nn.functional.softmax(alighment_scores, dim=2)

        # Calculate context vector
        # context shape: (batch_size, 1, hidden_dim)
        context = torch.bmm(attn_weights, encoder_outputs)

        # Concatenate output and context
        # output shape: (batch_size, 1, hidden_dim * 2)
        output = torch.cat((output, context), dim=2)

        # Final output projection
        # output shape: (batch_size, output_dim)
        output = self.fc_out(output.squeeze(1))

        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Initialize the EncoderDecoder model.

        Args:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the EncoderDecoder model.

        Args:
            src (Tensor): The source sequence input.
                Shape: (batch_size, src_seq_len, input_dim)
            trg (Tensor): The target sequence.
                Shape: (batch_size, trg_seq_len)
            teacher_forcing_ratio (float): The probability of using teacher forcing.
                Default: 0.5

        Returns:
            Tensor: The output sequence predictions.
                Shape: (batch_size, trg_seq_len, output_dim)
        """
        batch_size, trg_len = trg.shape

        # Initialize output tensor to store decoder outputs
        # Shape: (batch_size, trg_seq_len, output_dim)
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(src.device)

        # Pass the source sequence through the encoder
        # encoder_outputs shape: (batch_size, src_seq_len, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        encoder_outputs, hidden = self.encoder(src)

        # First input to the decoder is the first token of the target sequence
        input = trg[:, 0].unsqueeze(1) # dimension: batch size x len x hidden_in == batch size x 1 x 1
        # input shape: (batch_size, 1, output_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)

        for t in range(trg_len):
            # Pass the input, previous hidden state, and encoder outputs to the decoder
            # output shape: (batch_size, output_dim)
            # hidden shape: (num_layers, batch_size, hidden_dim)
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)

            # Store the output in the outputs tensor
            outputs[:, t] = output

            # During training, instead of using the model's own predictions as input for the next time step,
            # teacher forcing provides the ground truth (actual) output from the training data as input
            # Decide whether to use teacher forcing for the next input
            teacher_force = torch.rand(1) < teacher_forcing_ratio

            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = trg[:, t].unsqueeze(1) # if teacher_force else output.argmax(1)

        return outputs
