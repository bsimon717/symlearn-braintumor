import kagglehub

# Download latest version
path = kagglehub.dataset_download("mylee77/brain-tumor-mri-deduplicated-clean-version", output_dir="./data")

print("Path to dataset files:", path)