from openai import OpenAI
from PIL import Image
import os
import base64
import argparse
import glob
import json


TEMP_PATH = 'temp'


def create_temp():
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_file(path):
    os.remove(path)
    

def resize_and_convert_to_base64(image_path, max_size=(768, 768), image_extension='PNG'):
    # Open the image file
    with Image.open(image_path) as img:
        temp_path = TEMP_PATH + '/temp_' + image_path.split('/')[1]
        # Calculate the new size preserving the aspect ratio
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Save the image
        img.save(temp_path, format=image_extension)

        with open(temp_path, "rb") as resized_image_file:
            remove_temp_file(temp_path)
            return base64.b64encode(resized_image_file.read()).decode('utf-8')
        

def create_metadata_jsonl(image_paths, texts, data_focus='statue', output_file='data/metadata.jsonl'):
    """
    Creates a metadata.jsonl file from lists of image paths and texts.
    
    :param image_paths: List of image file paths.
    :param texts: List of text descriptions corresponding to the images.
    :param output_file: The output file name (default is 'data/metadata.jsonl').
    """
    if len(image_paths) != len(texts):
        raise ValueError("The length of image paths and texts must be the same.")
    
    caption_prefix = 'a photo of CUS ' + data_focus + ', '
    with open(output_file, 'w') as f:
        for image_path, text in zip(image_paths, texts):
            # Create a dictionary for each pair
            data = {
                "file_name": image_path,
                "text": caption_prefix + text
            }
            # Write the dictionary as a JSON object in a new line
            f.write(json.dumps(data) + '\n')


def annotate_image(image_path, image_extension, openai_secret_key=None):
    """
    Annotate an image using GPT-4.
    """
    image_base64 = resize_and_convert_to_base64(image_path, image_extension=image_extension)
    description = """Directly describe with brevity and as brief as possible the provided image.
    without any introductory phrase like 'This image shows', 'In the scene',
    'This image depicts' or similar phrases. Just start describing the scene please. Do not end the caption with a '.'
    Good examples: a cat on a windowsill, a photo of smiling cactus in an office, a man and baby sitting by a window, a photo of wheel on a car.
    """

    secret_key = os.environ.get("OPENAI_SECRET_KEY")
    if secret_key is None and openai_secret_key is not None:
        secret_key = openai_secret_key
    else:
        raise ValueError("No OpenAI secret key found. Please set the OPENAI_SECRET_KEY environment variable or pass it as an argument.")

    client = OpenAI(
        api_key=secret_key,
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": description},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        model="gpt-4o",
    )
    
    return response.choices[0].message.content


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Annotate data')

    # Add the arguments
    parser.add_argument('--images_path', type=str, required=True, help='Path to the images.')
    parser.add_argument('--image_extension', type=str, required=True, help='Extension of the images.')
    parser.add_argument('--secret_key', type=str, help='Secret key as a string.')
    parser.add_argument('--focus', type=str, help='Type the main object in the dataset. Like statue or kid')

    # Parse the arguments
    args = parser.parse_args()

    create_temp()
    annotations = []
    image_paths = []

    dataset_path =args.images_path + '/*.' + args.image_extension
    for path in glob.glob(dataset_path):
        annotated_image = annotate_image(path, args.image_extension.upper(), openai_secret_key=None if args.secret_key is None else args.secret_key)
        annotations.append(annotated_image)
        image_paths.append(path)

    # Create metadata
    create_metadata_jsonl(image_paths, annotations, data_focus=args.focus, output_file=args.images_path + '/metadata.json')


if __name__ == "__main__":
    main()
