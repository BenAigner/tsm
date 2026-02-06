import torch
from models.tsm_cnn_old import TSM_CNN

def main():

    # Configure input tensor dimensions

    N = 2  # Batch size
    T = 4  # Number of temporal segments
    C = 3  # Number of channels (RGB)
    H = W = 32  # Height and Width of CIFAR-10 images

    # Create a random input tensor
    input_tensor = torch.randn(N, T, C, H, W)

    # Initialize the TSM_CNN model
    model = TSM_CNN(num_classes=10, n_segment=T)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)


    # Shape assertions
    print("Input shape:", input_tensor.shape)   # Should be (N, T, 3, 32, 32)
    print("Output shape:", output.shape)        # should be (N, 10)

    # Check output dimensions
    assert output.shape == (N, 10), f"Expected output shape {(N, 10)}, but got {output.shape}"
    print("TSM_CNN forward pass successful. Output shape:", output.shape)

if __name__ == "__main__":
    main()