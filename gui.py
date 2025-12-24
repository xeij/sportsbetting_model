"""
Simple GUI for Football Betting Model

A clean, user-friendly interface for running the betting model.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import sys
import os
from pathlib import Path

# Import main functions
from main import download_data, train_models, run_backtest, fetch_live_odds, predict_fixtures


class BettingModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Betting Model")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e1e')
        
        # Style configuration
        self.setup_styles()
        
        # Create main layout
        self.create_widgets()
        
        # API key storage
        self.api_key = tk.StringVar()
        
    def setup_styles(self):
        """Configure ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Button style
        style.configure('Action.TButton',
                       background='#0d7377',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'),
                       padding=10)
        style.map('Action.TButton',
                 background=[('active', '#14a085')])
        
        # Label style
        style.configure('Title.TLabel',
                       background='#1e1e1e',
                       foreground='#ffffff',
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Info.TLabel',
                       background='#1e1e1e',
                       foreground='#b0b0b0',
                       font=('Segoe UI', 9))
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Header
        header_frame = tk.Frame(self.root, bg='#1e1e1e')
        header_frame.pack(fill='x', padx=20, pady=20)
        
        title = ttk.Label(header_frame, text="⚽ Football Betting Model", style='Title.TLabel')
        title.pack()
        
        subtitle = ttk.Label(header_frame, 
                            text="ML-powered value bet identification for English football",
                            style='Info.TLabel')
        subtitle.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Actions
        left_panel = tk.Frame(main_frame, bg='#2d2d2d', relief='flat')
        left_panel.pack(side='left', fill='both', padx=(0, 10), pady=0)
        
        actions_label = ttk.Label(left_panel, text="Actions", style='Title.TLabel')
        actions_label.pack(pady=15)
        
        # Action buttons
        self.create_action_button(left_panel, "📥 Download Data", self.download_data_action,
                                  "Download historical match data (16 seasons)")
        
        self.create_action_button(left_panel, "🤖 Train Models", self.train_models_action,
                                  "Train XGBoost & LightGBM models")
        
        self.create_action_button(left_panel, "📊 Run Backtest", self.run_backtest_action,
                                  "Simulate betting strategy on historical data")
        
        self.create_action_button(left_panel, "🔍 Fetch Live Odds", self.fetch_odds_action,
                                  "Get current odds from The Odds API")
        
        self.create_action_button(left_panel, "💰 Find Value Bets", self.predict_action,
                                  "Identify profitable betting opportunities")
        
        # API Key section
        api_frame = tk.Frame(left_panel, bg='#2d2d2d')
        api_frame.pack(fill='x', padx=15, pady=20)
        
        api_label = ttk.Label(api_frame, text="Odds API Key:", style='Info.TLabel')
        api_label.pack(anchor='w')
        
        self.api_entry = tk.Entry(api_frame, textvariable=self.api_key, 
                                 bg='#3d3d3d', fg='white', 
                                 insertbackground='white', relief='flat',
                                 font=('Consolas', 9))
        self.api_entry.pack(fill='x', pady=5)
        
        api_help = ttk.Label(api_frame, text="Get free key at the-odds-api.com", 
                           style='Info.TLabel', cursor='hand2')
        api_help.pack(anchor='w')
        api_help.bind('<Button-1>', lambda e: self.open_url('https://the-odds-api.com/'))
        
        # Right panel - Output
        right_panel = tk.Frame(main_frame, bg='#2d2d2d')
        right_panel.pack(side='right', fill='both', expand=True)
        
        output_label = ttk.Label(right_panel, text="Output", style='Title.TLabel')
        output_label.pack(pady=15)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            right_panel,
            bg='#1e1e1e',
            fg='#00ff00',
            insertbackground='white',
            font=('Consolas', 9),
            relief='flat',
            wrap='word'
        )
        self.output_text.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", 
                                   bg='#0d7377', fg='white',
                                   font=('Segoe UI', 9), anchor='w', padx=10)
        self.status_bar.pack(side='bottom', fill='x')
        
        # Redirect stdout to output text
        sys.stdout = TextRedirector(self.output_text, "stdout")
        
    def create_action_button(self, parent, text, command, tooltip):
        """Create a styled action button with tooltip."""
        btn_frame = tk.Frame(parent, bg='#2d2d2d')
        btn_frame.pack(fill='x', padx=15, pady=5)
        
        btn = ttk.Button(btn_frame, text=text, command=command, style='Action.TButton')
        btn.pack(fill='x')
        
        # Tooltip
        tip = ttk.Label(btn_frame, text=tooltip, style='Info.TLabel')
        tip.pack(anchor='w', pady=(2, 0))
        
    def download_data_action(self):
        """Download historical data."""
        self.run_in_thread(download_data, "Downloading data...")
        
    def train_models_action(self):
        """Train models."""
        self.run_in_thread(train_models, "Training models...")
        
    def run_backtest_action(self):
        """Run backtesting."""
        self.run_in_thread(run_backtest, "Running backtest...")
        
    def fetch_odds_action(self):
        """Fetch live odds."""
        api_key = self.api_key.get().strip()
        if not api_key:
            messagebox.showwarning("API Key Required", 
                                 "Please enter your Odds API key.\n\nGet a free key at:\nhttps://the-odds-api.com/")
            return
        
        self.run_in_thread(lambda: fetch_live_odds(api_key=api_key), "Fetching live odds...")
        
    def predict_action(self):
        """Make predictions."""
        self.run_in_thread(predict_fixtures, "Finding value bets...")
        
    def run_in_thread(self, func, status_message):
        """Run a function in a separate thread to prevent GUI freezing."""
        self.status_bar.config(text=status_message)
        self.output_text.delete(1.0, tk.END)
        
        def worker():
            try:
                func()
                self.status_bar.config(text="Complete ✓")
            except Exception as e:
                self.status_bar.config(text=f"Error: {str(e)}")
                print(f"\n[ERROR] {str(e)}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
    def open_url(self, url):
        """Open URL in browser."""
        import webbrowser
        webbrowser.open(url)


class TextRedirector:
    """Redirect stdout to a text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.update_idletasks()

    def flush(self):
        pass


def main():
    """Launch the GUI."""
    root = tk.Tk()
    app = BettingModelGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
