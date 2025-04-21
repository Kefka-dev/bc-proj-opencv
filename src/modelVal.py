from ultralytics import YOLO

# Load the pre-trained YOLO model you want to test
# Replace "yolov8m.pt" with the name or path of the pre-trained model file
# The ultralytics library can often download standard models by name
model = YOLO("../yolos/yolo11m.pt")

# Define the path to your data.yaml file
data_yaml_path = "/home/patrik/Documents/bc-proj-opencv/merged_dataset2/data.yaml" # Your data.yaml path

# Run validation mode specifically on the test split of your dataset
print("Evaluating pre-trained model on test data...")
metrics = model.val(data=data_yaml_path, split="test", plots=True, device=0, classes=[0])

# Print the evaluation metrics
print("\nTest Metrics (Pre-trained Model):")
# print(metrics)

# Access specific metrics
if metrics:
    print(metrics.box.maps)
    # print(f"Precision on Test Data: {metrics.results_dict.get('metrics/precision', 'N/A')}")
    # You can access other metrics as needed