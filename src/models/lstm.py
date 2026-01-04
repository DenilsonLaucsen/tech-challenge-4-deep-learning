import torch.nn as nn
from app.model.lstm_params import LSTMParams


class LSTM(nn.Module):
    """
    Flexible LSTM model composed from a list of layers created by the factory.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: (batch, seq_len, features) if batch_first=True
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
                # if next layer is not LSTM, reduce sequence dim and pass last time-step
                if i + 1 < len(self.layers) and not isinstance(
                    self.layers[i + 1], nn.LSTM
                ):
                    x = x[:, -1, :]  # take last time-step features
            else:
                x = layer(x)
        return x


class LSTMFactory:
    """
    Factory to create LSTM architectures given a layer_config and LSTMParams.
    layer_config can be an ordered dict or list of strings like ["LSTM", "Linear"].
    """

    def __init__(self, layer_config, params: LSTMParams):
        self.layer_config = (
            list(layer_config)
            if isinstance(layer_config, (list, tuple))
            else list(layer_config.values())
        )
        self.params = params

    def get_layer(self, layer_name: str, input_size=None):
        if input_size is None:
            input_size = self.params.input_size

        name = layer_name.lower()
        if name == "lstm":
            return nn.LSTM(
                input_size=input_size,
                hidden_size=self.params.hidden_size,
                num_layers=self.params.num_layers,
                batch_first=self.params.batch_first,
                dropout=self.params.dropout if self.params.num_layers > 1 else 0.0,
            )
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "linear":
            return nn.Linear(input_size, self.params.output_size)
        if name == "flatten":
            return nn.Flatten()
        raise ValueError(f"Layer {layer_name} not supported by LSTMFactory")

    def create(self):
        layers = []
        current_input_size = self.params.input_size
        for layer_type in self.layer_config:
            # if layer is LSTM, create with current_input_size and update
            if layer_type.lower() == "lstm":
                layer = self.get_layer("LSTM", input_size=current_input_size)
                # output of LSTM has size hidden_size across features
                current_input_size = self.params.hidden_size
            elif layer_type.lower() == "linear":
                layer = self.get_layer("Linear", input_size=current_input_size)
                current_input_size = self.params.output_size
            else:
                layer = self.get_layer(layer_type, input_size=current_input_size)
            layers.append(layer)
        return LSTM(layers)
