import tritonclient.http as httpclient
import numpy as np
import cv2
import os
import argparse


def get_label_from_logits(logits):
    """Convert raw logits into a class label."""
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    label = np.argmax(probabilities, axis=-1)  # Get the index of the highest probability
    return label, probabilities

def preprocess_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    return np.expand_dims(image, axis=0)  # Add batch dimension

def perform_inference(client, input_data, model_name="age_model"):
    inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [httpclient.InferRequestedOutput("output")]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return response.as_numpy("output")

def infer_on_image(client, image_path):
    input_data = preprocess_image(image_path)
    logits = perform_inference(client, input_data)
    label, probabilities = get_label_from_logits(logits)
    print(f"Inference result for {image_path}: Label={label}, Probabilities={probabilities}")

def infer_on_folder(client, folder_path):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                print(f"Processing {img_path}...")
                input_data = preprocess_image(img_path)
                logits = perform_inference(client, input_data)
                label, probabilities = get_label_from_logits(logits)
                print(f"Inference result for {img_path}: Label={label}, Probabilities={probabilities}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Inference Script")
    parser.add_argument("--img", type=str, help="Path to a single image.")
    parser.add_argument("--folder", type=str, help="Path to a folder containing images.")
    parser.add_argument("--url", type=str, default="localhost:8000", help="Triton server URL.")
    parser.add_argument("--model_name", type=str, default="age_model", help="Model name on Triton server.")

    args = parser.parse_args()

    # Create Triton client
    client = httpclient.InferenceServerClient(url=args.url)

    # Perform inference based on the provided arguments
    if args.img:
        infer_on_image(client, args.img)
    elif args.folder:
        infer_on_folder(client, args.folder)
    else:
        print("Please specify --img or --folder.")
