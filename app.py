"""
Hugging Face Spaces Entry Point for Football Betting Model

This is the main entry point that Hugging Face Spaces will use.
It's identical to gradio_app.py but named app.py for HF compatibility.
"""

# Import everything from the gradio app
from gradio_app import *

if __name__ == "__main__":
    # Create and launch the interface for Hugging Face Spaces
    demo = create_interface()
    demo.launch(
        theme=gr.themes.Base(),
        css=css
    )