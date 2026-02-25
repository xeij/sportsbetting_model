"""
Gradio Web Interface for Football Betting Model

A modern web interface that replicates the tkinter GUI functionality.
"""

import gradio as gr
import pandas as pd
import sys
import io
import traceback
import threading
import time
import queue
from contextlib import redirect_stdout, redirect_stderr

# Import main functions
from main import download_data, train_models, run_backtest, fetch_live_odds, predict_fixtures


class StreamingOutput:
    """Real-time output streaming for Gradio."""

    def __init__(self):
        self.output_queue = queue.Queue()
        self.current_output = ""
        self.is_running = False

    def write(self, text):
        """Write method for stdout redirection."""
        if text and text.strip():
            self.output_queue.put(text)

    def flush(self):
        """Flush method for stdout redirection."""
        pass

    def get_current_output(self):
        """Get all current output."""
        while not self.output_queue.empty():
            try:
                text = self.output_queue.get_nowait()
                self.current_output += text
            except queue.Empty:
                break
        return self.current_output

    def clear_output(self):
        """Clear the current output."""
        self.current_output = ""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def run_with_streaming(self, func, *args, **kwargs):
        """Run a function with real-time output streaming."""
        self.clear_output()
        self.is_running = True

        def worker():
            try:
                # Redirect stdout to our streaming output
                original_stdout = sys.stdout
                sys.stdout = self

                # Run the function
                result = func(*args, **kwargs)

                # Restore stdout
                sys.stdout = original_stdout

                # Mark as complete
                self.output_queue.put("\n" + "=" * 80 + "\n")
                self.output_queue.put("OPERATION COMPLETE\n")
                self.output_queue.put("=" * 80 + "\n")

            except Exception as e:
                # Restore stdout
                sys.stdout = original_stdout
                error_msg = f"\n[ERROR] {str(e)}\n{traceback.format_exc()}"
                self.output_queue.put(error_msg)
            finally:
                self.is_running = False

        # Start the worker thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        return thread

# Global streaming output
streaming_output = StreamingOutput()


def download_data_action():
    """Download historical data with streaming output."""
    streaming_output.run_with_streaming(download_data)
    return streaming_output.get_current_output(), "🔄 Processing..."


def train_models_action():
    """Train models with streaming output."""
    streaming_output.run_with_streaming(train_models)
    return streaming_output.get_current_output(), "🔄 Processing..."


def run_backtest_action():
    """Run backtesting with streaming output."""
    streaming_output.run_with_streaming(run_backtest)
    return streaming_output.get_current_output(), "🔄 Processing..."


def fetch_odds_action(api_key):
    """Fetch live odds with streaming output."""
    if not api_key or not api_key.strip():
        error_msg = """[ERROR] API Key Required

Please enter your Odds API key.

Get a free key at: https://the-odds-api.com/
        """
        return error_msg, "✗ API Key Required"

    streaming_output.run_with_streaming(fetch_live_odds, api_key=api_key.strip())
    return streaming_output.get_current_output(), "🔄 Processing..."


def predict_action():
    """Make predictions with streaming output."""
    streaming_output.run_with_streaming(predict_fixtures)
    return streaming_output.get_current_output(), "🔄 Processing..."


def update_output():
    """Update the output display with latest content."""
    current_output = streaming_output.get_current_output()
    if streaming_output.is_running:
        status = "🔄 Processing..."
    else:
        status = "✓ Complete" if current_output and not "[ERROR]" in current_output else "Ready"

    return current_output, status


def check_model_status():
    """Check model status on startup."""
    try:
        from src.model_checker import check_model_freshness

        status = check_model_freshness()

        output_lines = [
            "=" * 80,
            "MODEL STATUS CHECK",
            "=" * 80,
            ""
        ]

        if not status['data_exists']:
            output_lines.extend([
                "[WARNING] No data found.",
                "Click 'Download Data' to get started.",
                ""
            ])
            status_msg = "No data - Download required"
        elif not status['models_exist']:
            output_lines.extend([
                "[WARNING] No trained models found.",
                "Click 'Train Models' after downloading data.",
                ""
            ])
            status_msg = "No models - Training required"
        elif not status['models_up_to_date']:
            output_lines.extend([
                "[WARNING] Models are outdated!",
                f"Data updated: {status['data_date'].strftime('%Y-%m-%d %H:%M')}",
                f"Models trained: {status['model_date'].strftime('%Y-%m-%d %H:%M')}",
                "",
                "Click 'Train Models' to retrain with latest data.",
                ""
            ])
            status_msg = "Models outdated - Retraining recommended"
        else:
            output_lines.extend([
                "[OK] Models are up to date.",
                f"Trained: {status['model_date'].strftime('%Y-%m-%d %H:%M')}",
                "",
                "Ready to run backtests or find value bets.",
                ""
            ])
            status_msg = "Ready - Models up to date"

        output_lines.append("=" * 80)

        return "\n".join(output_lines), status_msg

    except Exception as e:
        return f"[INFO] Could not check model status: {str(e)}", "Ready"


# Custom CSS for dark theme
css = """
/* Main container - pure black background */
.gradio-container {
    background-color: #000000 !important;
    color: #ffffff !important;
}

/* Overall app background */
.app {
    background-color: #000000 !important;
}

/* Blocks container */
.block {
    background-color: #000000 !important;
}

/* Buttons - dark gray like original tkinter */
.gr-button {
    background-color: #404040 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    margin: 5px !important;
    font-weight: normal !important;
    padding: 10px 20px !important;
}

.gr-button:hover {
    background-color: #505050 !important;
}

/* Output terminal - black background with bright green text */
.gr-textbox textarea {
    background-color: #000000 !important;
    color: #00ff00 !important;
    border: 1px solid #404040 !important;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    font-size: 13px !important;
}

/* API key input - dark gray like original */
.gr-textbox input {
    background-color: #2d2d2d !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
}

/* Panels and containers - dark gray */
.gr-panel {
    background-color: #1a1a1a !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
}

.gr-form {
    background-color: #1a1a1a !important;
}

.gr-box {
    background-color: #1a1a1a !important;
    border-radius: 8px !important;
}

/* Column backgrounds */
.gr-column {
    background-color: #1a1a1a !important;
}

/* Markdown text styling */
.markdown-body {
    color: #cccccc !important;
    background-color: transparent !important;
}

.markdown-body h1, .markdown-body h2, .markdown-body h3 {
    color: #ffffff !important;
}

/* Status display */
.status-display {
    background-color: #2d2d2d !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
}

/* Remove any unwanted borders or backgrounds */
.gr-row {
    background-color: transparent !important;
}
"""


def create_interface():
    """Create the Gradio interface."""

    # Initialize status
    initial_output, initial_status = check_model_status()

    with gr.Blocks(title="Football Betting Model") as demo:
        # Header
        gr.Markdown("""
        # Football Betting Model
        ## Machine learning value bet identification
        """)

        with gr.Row():
            # Left Panel - Actions
            with gr.Column(scale=1):
                gr.Markdown("### Actions")

                # Action buttons
                download_btn = gr.Button("Download Data")
                gr.Markdown("*Download historical match data*")

                train_btn = gr.Button("Train Models")
                gr.Markdown("*Train XGBoost and LightGBM models*")

                backtest_btn = gr.Button("Run Backtest")
                gr.Markdown("*Simulate betting strategy*")

                odds_btn = gr.Button("Fetch Live Odds")
                gr.Markdown("*Get current odds from API*")

                predict_btn = gr.Button("Find Value Bets")
                gr.Markdown("*Identify betting opportunities*")

                # API Key section
                gr.Markdown("### Odds API Key")
                api_key_input = gr.Textbox(
                    label="",
                    placeholder="Enter your API key...",
                    type="password",
                    lines=1,
                    elem_classes=["api-key-input"]
                )
                gr.Markdown("[Get free key at the-odds-api.com](https://the-odds-api.com/)")

            # Right Panel - Output
            with gr.Column(scale=2):
                gr.Markdown("### Output")

                output_display = gr.Textbox(
                    value=initial_output,
                    label="",
                    lines=25,
                    max_lines=25,
                    interactive=False
                )

        # Status bar
        status_display = gr.Textbox(
            value=initial_status,
            label="Status",
            lines=1,
            interactive=False
        )

        # Auto-refresh timer for live updates
        refresh_timer = gr.Timer(0.5, active=False)  # Update every 500ms when active

        # Button click handlers with auto-refresh
        def start_download():
            result = download_data_action()
            refresh_timer.active = True  # Start auto-refresh
            return result

        def start_training():
            result = train_models_action()
            refresh_timer.active = True  # Start auto-refresh
            return result

        def start_backtest():
            result = run_backtest_action()
            refresh_timer.active = True  # Start auto-refresh
            return result

        def start_odds_fetch(api_key):
            result = fetch_odds_action(api_key)
            if "API Key Required" not in result[1]:
                refresh_timer.active = True  # Start auto-refresh only if no error
            return result

        def start_prediction():
            result = predict_action()
            refresh_timer.active = True  # Start auto-refresh
            return result

        def update_and_check_completion():
            """Update output and stop timer if operation is complete."""
            output, status = update_output()
            # Stop the timer if operation is complete
            if not streaming_output.is_running:
                refresh_timer.active = False
            return output, status

        # Connect buttons
        download_btn.click(
            fn=start_download,
            outputs=[output_display, status_display]
        )

        train_btn.click(
            fn=start_training,
            outputs=[output_display, status_display]
        )

        backtest_btn.click(
            fn=start_backtest,
            outputs=[output_display, status_display]
        )

        odds_btn.click(
            fn=start_odds_fetch,
            inputs=[api_key_input],
            outputs=[output_display, status_display]
        )

        predict_btn.click(
            fn=start_prediction,
            outputs=[output_display, status_display]
        )

        # Connect timer for live updates
        refresh_timer.tick(
            fn=update_and_check_completion,
            outputs=[output_display, status_display]
        )

    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=gr.themes.Base(),
        css=css
    )