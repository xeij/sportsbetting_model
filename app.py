"""
Hugging Face Spaces Entry Point for Football Betting Model

This is the main entry point that Hugging Face Spaces will use.
"""

import gradio as gr
from gradio_app import create_interface, css

if __name__ == "__main__":
    # Create and launch the interface for Hugging Face Spaces
    demo = create_interface()
    demo.launch(
        theme=gr.themes.Base(),
        css=css
    )