# AN2DL 2023 - Homework 2

## Dataset Specification

For this assignment, we were provided with a dataset containing **48,000 time series**, each of varying lengths but zero-padded to a common length of **2,776 time steps**. Along with the data, we received:

- **Valid Periods**: An array specifying, for each time series, its start and end index (i.e., the actual data without padding).
- **Categories**: An array indicating, for each time series, its category code (one of A, B, C, D, E, or F).

### Dataset Pre-Processing

Our initial approach involved a straightforward exploration of the dataset:

- **Normalization**: Observed that all time series were already normalized within the [0, 1] range.
- **Outlier Detection**: Identified numerous outliers across the time series. Attempted **Robust Scaling** to mitigate their impact, but it did not enhance model performance and was thus discarded.

### The Padding Problem

Given that the time series were zero-padded on the left:

- **No Padding Approach**: Removing zero padding and using the series as-is did not yield favorable results.
- **Interpolation Padding**: Replaced zeros with a linear interpolation of missing values for each time series. Pre-padding was chosen based on its effectiveness with LSTM models [^1].

### Training and Validation Sets

Considering the testing would occur on a server, we split our dataset into **training and validation sets**, experimenting with validation splits ranging from **5% to 10%**.

## Framework and Sequence Generation

We adopted an **autoregression framework**, feeding the model with data obtained by sliding a window of fixed size over the input data. Specifically, we used a **Multi-Input/Multi-Output** strategy:

- **Input**: Window of \( x_1, ..., x_t \)
- **Output**: Values \( x_{t+1}, ..., x_{t+N} \)

To generate sequences:

- Developed a custom function to extract subsequences with specified length, future window size, and stride.
- Determined optimal parameters:
  - **Sequence Length**: 200
  - **Stride**: 50
  - **Future Window Size**: 18

Batching was performed by grouping **60 time series** together, reducing training time.

## Development

Our development process involved multiple modeling approaches:

### Base Models

Tested basic models to establish a performance baseline:

- **RNN**
- **LSTM**
- **Stacked-LSTM**

All models utilized **512 recurrent units** for fair comparison.

### Convolutions and LSTMs

Combined **Convolutional Neural Networks (CNN)** with **LSTM** architectures to leverage both local and temporal patterns:

- **Bidirectional ConvLSTM**: A Bidirectional LSTM layer followed by two 1D Convolution layers.
- **ResNet Model**
- **WaveNet-Inspired Model**: Utilized **dilated causal convolutions** as per DeepMind's WaveNet [^2].

The **Bidirectional ConvLSTM** outperformed other architectures in this category.

### Seq2Seq and Attention

Implemented a **Sequence-to-Sequence (Seq2Seq)** model with an **Encoder-Decoder** structure:

- Both encoder and decoder utilized LSTM layers, with one being bidirectional.
- Enhanced with the **Luong Attention Mechanism** [^3].

This model did not surpass the performance of the **Bidirectional ConvLSTM**.

### Multiple Models

Explored creating separate models for each category:

- Trained individual models per class.
- Also trained the base model on all data but validated on specific classes.

This approach did not yield better results compared to the general **Bidirectional ConvLSTM** model.

## Final Model, Parameters, and Results

The chosen final model is the **Bidirectional ConvLSTM**, structured as follows:

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1qtTPLOLixGq3GFMyOBWptCci3hdscmth" alt="Bidirectional ConvLSTM Architecture" width="50%">
</p>

**Hyperparameters:**

- **Input Window Size**: 200
- **Output Window Size**: 18
- **Learning Rate**: 1e-3 (with reduction on plateau up to 1e-7)
- **LSTM Units**: 512

**Performance Metrics:**
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1uQCVrJfQrCmCxyBTfB8lnm8I_3374qHC" alt="Performance Metrics" width="50%">
</p>
---

[^1]: *Reference for Pre-padding*: [Pre-padding in LSTM Models](https://arxiv.org/abs/1503.04069)

[^2]: *WaveNet Paper*: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

[^3]: *Luong Attention*: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
