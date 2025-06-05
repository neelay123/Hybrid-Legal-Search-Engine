import tkinter as tk
from tkinter import ttk, messagebox
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import numpy as np
import logging
import threading
import re  # Make sure to import re

# --------------------------- Configuration ---------------------------
CONFIG = {
    "dataset_name": "coastalcph/lex_glue",
    "subsets": {
        "CaseHOLD": ("case_hold", "context"),
        "ECtHR A": ("ecthr_a", "facts"),
        "ECtHR B": ("ecthr_b", "facts"),
        "EUR-Lex": ("eurlex", "text"),
        "SCOTUS": ("scotus", "text")
    },
    "default_results": 10,
    "colors": {
        "primary": "#2C3E50",
        "secondary": "#3498DB",
        "background": "#ECF0F1",
        "text": "#2C3E50",
        "highlight": "#E74C3C"
    },
    "font": ("Helvetica", 12)
}


# --------------------------- Search Engine Core ---------------------------
class LegalSearchEngine:
    def __init__(self):
        self.dataset = None
        self.bm25 = None
        self.current_subset = None
        self.text_column = None

    def load_dataset(self, subset_name):
        """Load dataset with comprehensive error handling and fallback for text column."""
        try:
            if subset_name not in CONFIG["subsets"]:
                raise ValueError("Invalid dataset selection")

            # Clear previous data
            self.dataset = None
            self.bm25 = None

            subset_key, text_col = CONFIG["subsets"][subset_name]
            self.current_subset = subset_key
            self.text_column = text_col

            # Load from Hugging Face Hub
            dataset = load_dataset(
                CONFIG["dataset_name"],
                subset_key,
                trust_remote_code=True
            )

            # Convert to pandas DataFrame
            self.dataset = dataset["train"].to_pandas()

            # Validate dataset structure: check if the expected text column exists
            if self.text_column not in self.dataset.columns:
                # Fallback mechanism for ECtHR datasets:
                # If the current subset is one of the ECtHR datasets and alternative 'text' column exists, use it.
                if self.current_subset in ("ecthr_a", "ecthr_b") and "text" in self.dataset.columns:
                    logging.warning(
                        f"Expected column '{self.text_column}' not found; using 'text' instead for {subset_name}.")
                    self.text_column = "text"
                else:
                    raise ValueError(f"Text column '{self.text_column}' not found")

            if self.dataset.empty:
                raise ValueError("Loaded dataset is empty")

            # Build search index
            corpus = self.dataset[self.text_column].fillna("").astype(str).tolist()
            self.bm25 = BM25Okapi([doc.split() for doc in corpus])

            return True

        except Exception as e:
            logging.error(f"Dataset error: {str(e)}")
            self.dataset = None
            self.bm25 = None
            messagebox.showerror(
                "Load Error",
                f"Failed to load {subset_name}:\n{str(e)}\n\n"
                "Possible solutions:\n"
                "1. Check internet connection\n"
                "2. Verify Hugging Face dataset availability\n"
                "3. Try a different dataset subset"
            )
            return False

    def search(self, query):
        """Execute search with validation"""
        if self.bm25 is None or self.dataset is None:
            raise ValueError("Search index not initialized")

        # BM25 retrieval
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-CONFIG["default_results"]:][::-1]

        # Format results
        return [{
            "text": self._highlight_query(query, self.dataset.iloc[idx][self.text_column]),
            "score": f"{doc_scores[idx]:.2f}",
            "source": self.current_subset
        } for idx in top_indices]

    def _highlight_query(self, query, text):
        # If text is an array or a list, join its elements into a single string.
        if isinstance(text, (list, np.ndarray)):
            text = " ".join(map(str, text))

        # Ensure that text is a string.
        text = str(text)
        lower_text = text.lower()

        # Split query into terms (case-insensitively).
        query_terms = query.lower().split()

        # Find all occurrences of each query term in the text.
        spans = []
        for term in query_terms:
            start = 0
            while True:
                idx = lower_text.find(term, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(term)))
                start = idx + len(term)

        # Sort the spans and merge any overlapping ones.
        spans.sort()
        merged = []
        for span in spans:
            if not merged or span[0] > merged[-1][1]:
                merged.append(span)
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))

        # Build the highlighted text by inserting markers for each found span.
        highlighted = []
        last_pos = 0
        for start, end in merged:
            highlighted.append(text[last_pos:start])
            highlighted.append(f"«{text[start:end]}»")
            last_pos = end
        highlighted.append(text[last_pos:])

        return "".join(highlighted)


# --------------------------- Modern GUI Interface ---------------------------
class LegalSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.engine = LegalSearchEngine()
        self.title("Legal Search")
        self.geometry("1280x800")
        self.configure(bg=CONFIG["colors"]["background"])
        self._setup_styles()
        self._setup_ui()

    def _setup_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure("TFrame", background=CONFIG["colors"]["background"])
        style.configure("TLabel",
                        background=CONFIG["colors"]["background"],
                        foreground=CONFIG["colors"]["text"],
                        font=CONFIG["font"])
        style.configure("TButton",
                        background=CONFIG["colors"]["secondary"],
                        foreground="white",
                        font=CONFIG["font"],
                        borderwidth=1)
        style.map("TButton",
                  background=[("active", CONFIG["colors"]["primary"])])

        # Text widget tags
        self.text_tags = {
            "header": {"font": ("Helvetica", 14, "bold"), "foreground": CONFIG["colors"]["primary"]},
            "highlight": {"background": "#FFF3E0", "foreground": CONFIG["colors"]["highlight"],
                          "font": ("Helvetica", 12, "bold")},
            "separator": {"foreground": CONFIG["colors"]["secondary"], "font": ("Helvetica", 8)}
        }

    def _setup_ui(self):
        """Build professional interface"""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Section
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        ttk.Label(header_frame,
                  text="⚖ Legal Search Engine",
                  font=("Helvetica", 20, "bold"),
                  anchor=tk.CENTER).pack(fill=tk.X)

        # Dataset Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=20)

        ttk.Label(control_frame, text="Select Legal Domain:").pack(side=tk.LEFT, padx=5)
        self.subset_var = tk.StringVar()
        self.dataset_selector = ttk.Combobox(
            control_frame,
            textvariable=self.subset_var,
            values=list(CONFIG["subsets"].keys()),
            state="readonly",
            width=20
        )
        self.dataset_selector.pack(side=tk.LEFT, padx=5)
        self.dataset_selector.current(0)

        ttk.Button(control_frame,
                   text="Load Dataset",
                   command=self._load_dataset).pack(side=tk.LEFT, padx=10)

        # Search Section
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=10)

        self.search_entry = ttk.Entry(search_frame,
                                      font=CONFIG["font"],
                                      width=60)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(search_frame,
                   text="🔍 Search",
                   command=self._threaded_search).pack(side=tk.LEFT, padx=10)

        # Results Display
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(results_frame,
                                    wrap=tk.WORD,
                                    font=CONFIG["font"],
                                    bg="white",
                                    padx=10,
                                    pady=10)
        for tag, config in self.text_tags.items():
            self.results_text.tag_configure(tag, **config)

        scrollbar = ttk.Scrollbar(results_frame,
                                  orient=tk.VERTICAL,
                                  command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status Bar
        self.status_bar = ttk.Label(main_frame,
                                    text="Ready",
                                    relief=tk.SUNKEN,
                                    anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=5)

    def _clear_ui(self):
        """Reset search interface"""
        self.search_entry.delete(0, tk.END)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.status_bar.config(text="Ready")

    def _load_dataset(self):
        """Load dataset with visual feedback"""
        self._clear_ui()
        subset = self.subset_var.get()
        self.status_bar.config(text=f"Loading {subset}...")
        self.update_idletasks()

        try:
            if self.engine.load_dataset(subset):
                messagebox.showinfo("Success",
                                    f"Successfully loaded {subset} dataset with "
                                    f"{len(self.engine.dataset)} documents")
                self.status_bar.config(text=f"Loaded {subset}")
            else:
                self.status_bar.config(text="Dataset load failed")

        except Exception as e:
            messagebox.showerror("Critical Error", f"Unexpected error: {str(e)}")
            self.status_bar.config(text="Load failed - critical error")

    def _threaded_search(self):
        """Run search in background thread"""
        self.status_bar.config(text="Searching...")
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Searching...\n")
        self.results_text.config(state=tk.DISABLED)

        threading.Thread(target=self._perform_search).start()

    def _perform_search(self):
        """Execute and display search results"""
        try:
            query = self.search_entry.get()
            if not query:
                raise ValueError("Please enter a search query")

            if self.engine.dataset is None:
                raise ValueError("No dataset loaded")

            results = self.engine.search(query)

            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            for idx, result in enumerate(results):
                # Header with metadata
                self.results_text.insert(
                    tk.END,
                    f"MATCH {idx + 1} [Score: {result['score']}] [Source: {result['source'].upper()}]\n",
                    "header"
                )

                # Use regex-based splitting to insert highlighted text segments
                parts = re.split(r"(«.*?»)", result['text'])
                for part in parts:
                    if part.startswith("«") and part.endswith("»"):
                        # Insert the highlighted portion without the markers
                        self.results_text.insert(tk.END, part[1:-1], "highlight")
                    else:
                        self.results_text.insert(tk.END, part)

                self.results_text.insert(tk.END, "\n\n" + "━" * 100 + "\n\n", "separator")

            self.status_bar.config(text=f"Found {len(results)} results")

        except Exception as e:
            messagebox.showerror("Search Error", str(e))
            self.status_bar.config(text="Search failed")
        finally:
            self.results_text.config(state=tk.DISABLED)


# --------------------------- Main Execution ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app = LegalSearchApp()
    app.mainloop()
