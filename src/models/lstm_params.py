from pydantic import BaseModel, Field


class LSTMParams(BaseModel):
    """
    Pydantic model for LSTM hyperparameters.
    """

    input_size: int | None = Field(
        None, description="Number of features in the input data."
    )
    hidden_size: int = Field(
        ..., description="Number of hidden units in the LSTM layer(s)."
    )
    num_layers: int = Field(..., description="Number of stacked LSTM layers.")
    output_size: int = Field(
        ..., description="Number of output units (e.g., for regression)."
    )
    batch_first: bool = Field(
        True, description="If True, tensors are (batch, seq, feature)."
    )
    dropout: float = Field(0.0, description="Dropout between LSTM layers.")
