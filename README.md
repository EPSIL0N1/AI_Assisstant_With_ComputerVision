# AI Assisstant With Computer Vision

This project integrating visual perception and natural language processing using YOLO for object detection, Groq for language modeling, and various APIs for image captioning. The project is designed to engage in natural conversations, informed by its visual perception, to assist users in an interactive manner.

## Features

- **Visual Perception**: Utilizes YOLOv8 for real-time object detection.
- **Image Captioning**: Uses multiple APIs to generate captions for images.
- **Conversational AI**: Employs Groq's language model to engage in natural, friendly conversations with users.
- **Tool Integration**: Integrates tools like TavilySearchResults for retrieving additional information.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/EPSIL0N1/EPIC_Final_Project.git
    cd EPIC_Final_Project
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    Create a `.env` file in the root directory and add your Groq API key:

    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

### Running the Application

To run the application, execute the following command:

```sh
python your_script_name.py
```

Replace `your_script_name.py` with the name of your main script file.

### Command Line Arguments

- `--webcam-resolution`: Set the resolution of the webcam. Default is `[1280, 720]`.

### Example

```sh
python your_script_name.py --webcam-resolution 1920 1080
```

## Code Overview

### Object Detection and Image Captioning

The script captures frames from the webcam and uses YOLOv8 for object detection. It then sends the frames to multiple image captioning APIs to generate descriptions of the scenes.

### Conversational AI

The conversational part of the application is handled by Groq's language model, which processes user inputs and generates responses based on the visual observations and integrated tools.

### Main Functions

- `main1()`: Handles the webcam input, object detection, and image captioning.
- `main2()`: Manages the conversational AI, integrating the visual perception data into the chat responses.

### Example Code Snippet

```python
def main1():
    # Your code for object detection and image captioning
    pass

def main2():
    # Your code for conversational AI
    pass

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(main1)
        future2 = executor.submit(main2)
        future1.result()
        future2.result()
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.
