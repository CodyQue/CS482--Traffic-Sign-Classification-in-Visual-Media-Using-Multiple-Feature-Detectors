import os

# Path to the folder containing images
folder_path = "signs/"

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is an image (you may need to adjust the condition based on your file extensions)
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
        # Construct the full path to the image file
        image_file = os.path.join(folder_path, file_name)
        print(image_file)  # Or perform any other operation with the image file