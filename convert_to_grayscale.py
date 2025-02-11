from PIL import Image
import os

def convert_to_grayscale(input_path, output_path):
    try:
        # Open an image file
        img = Image.open(input_path)

        # Convert the image to grayscale
        img_gray = img.convert("L")

        # Save the converted image
        img_gray.save(output_path)
        print(f"Conversion successful: {input_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    # Assuming your image files are in a directory
    input_directory = "C:/Users/user/OneDrive/Desktop/data_p/surprise/"
    output_directory = "C:/Users/user/OneDrive/Desktop/data_p/surprise/converted/"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Convert the image to grayscale and save it
            convert_to_grayscale(input_path, output_path)

    print("Conversion complete.")
