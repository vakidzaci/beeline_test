FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Copy the model repository into the container
COPY model_repository /models

# Run Triton server with the model repository
CMD ["tritonserver", "--model-repository=/models", "--allow-gpu-metrics=true"]
