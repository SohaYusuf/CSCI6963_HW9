import os
import csv
from PIL import Image

def resize_images_and_create_csv(main_dir, output_dir, width, height, csv_path):
    """
    Resize all images in subdirectories, assign labels based on directory names, and create a CSV file.

    :param main_dir: Main directory containing subdirectories with images
    :param output_dir: Directory to save resized images
    :param width: Target width of the images
    :param height: Target height of the images
    :param csv_path: Path to save the CSV file
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the CSV file in append mode
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Iterate through all subdirectories in the main directory
        for subdir_name in os.listdir(main_dir):
            subdir_path = os.path.join(main_dir, subdir_name)

            # Check if it's a directory
            if os.path.isdir(subdir_path):
                # Assign labels based on subdirectory name
                if subdir_name == "87gym":
                    label = 0
                elif subdir_name == "Amos Eaton":
                    label = 1
                elif subdir_name == "EMPAC":
                    label = 2
                elif subdir_name == "Greene":
                    label = 3
                elif subdir_name == "JEC":
                    label = 4
                elif subdir_name == "Lally":
                    label = 5
                elif subdir_name == "Library":
                    label = 6
                elif subdir_name == "Ricketts":
                    label = 7
                elif subdir_name == "Sage":
                    label = 8
                elif subdir_name == "Troy Building":
                    label = 9
                elif subdir_name == "Voorhees":
                    label = 10
                else:
                    print(f"Skipping directory '{subdir_name}' (no label assigned).")
                    continue

                # Process all images in the subdirectory
                for file_name in os.listdir(subdir_path):
                    try:
                        input_path = os.path.join(subdir_path, file_name)

                        # Check if the file is an image
                        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                            # Open and resize the image
                            with Image.open(input_path) as img:
                                img_rgb = img.convert('RGB')
                                resized_img = img_rgb.resize((width, height), Image.Resampling.LANCZOS)
                                # Save the resized image in the output directory
                                output_image_path = os.path.join(output_dir, file_name)
                                resized_img.save(output_image_path)

                                # Append the image name and label to the CSV file
                                csv_writer.writerow([file_name, label])
                                print(f"Processed and resized: {file_name} (Label: {label})")
                        else:
                            print(f"Skipped non-image file: {file_name}")
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

# Example usage
main_directory = "hw9\CSCI6963_HW9-my-code\Custom_Data_Set"  # Replace with the path to your main directory
output_directory = "hw9\CSCI6963_HW9-my-code\Custom_Data_Set_Resized_v2"  # Replace with the path to your output directory
csv_file_path = "hw9\CSCI6963_HW9-my-code\Custom_Data_Set_Resized_v2\Custom_Data_Set_Labels.csv"  # Replace with your desired CSV file path

resize_images_and_create_csv(main_directory, output_directory, 252, 189, csv_file_path)
