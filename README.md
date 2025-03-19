# Sorting Algorithm Visualizer

A web-based sorting algorithm visualizer built using Flask, JSON requests in `server.py`, and an interactive front-end using HTML, CSS, and JavaScript.

## Features
- Visualizes various sorting algorithms (e.g., Bubble Sort, Merge Sort, Quick Sort, etc.).
- Uses Flask for the backend to handle sorting requests via JSON.
- Interactive UI with animations for sorting steps.
- Displays comparisons and time complexity analysis.
- Compares sorting performance with Yao's lower bound.

## Technologies Used
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Communication:** JSON requests between client and server

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sorting-visualizer.git
   cd sorting-visualizer
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install flask
   ```
4. Run the Flask server:
   ```bash
   python server.py
   ```
5. Open `index.html` in a browser or navigate to `http://127.0.0.1:5000/`


## Usage
1. Select a sorting algorithm from the UI.
2. Input or generate a random array.
3. Click "Sort" to visualize the process step-by-step.
4. Observe comparisons, swaps, and time complexity analysis.

## Contribution
Feel free to fork this repository, open issues, and contribute by submitting pull requests!
