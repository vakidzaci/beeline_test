import torch
from models import CombinedModel

combined_model = CombinedModel()
# Load the trained model
combined_model.load_state_dict(torch.load("../models/best_combined.pt"))
combined_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example input tensor (batch size=1, 3 color channels, image size 128x128)
dummy_input = torch.randn(1, 3, 128, 128)
# Export to ONNX
onnx_file_path = "../models/combined_model.onnx"
torch.onnx.export(
    combined_model,               # Model to export
    dummy_input,                  # Example input
    onnx_file_path,               # Path to save the ONNX file
    export_params=True,           # Store trained parameters
    opset_version=11,             # ONNX version
    input_names=["input"],        # Input layer names
    output_names=["output"],      # Output layer names
    dynamic_axes={                # Support dynamic batch sizes
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
print(f"Model exported to {onnx_file_path}")