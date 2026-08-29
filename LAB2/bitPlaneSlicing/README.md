# 🌐 Digital Image Processing WebApp

An interactive web application built with **Streamlit** and **OpenCV** to perform key digital image processing techniques:

- **Global Histogram Equalization (GHE)**
- **Local Histogram Equalization (LHE)**
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**
- **Bit Plane Slicing**

The app provides a modern, browser-based interface where you can upload images, apply transformations, and visualize results side by side.

---

## 📂 Project Structure

ImageProcessingWebApp/
│
├── app.py                           # Main Streamlit app
│
├── histogram_equalization/
│   ├── global_equalization.py       # GHE implementation
│   ├── local_equalization.py        # LHE implementation
│   └── clahe_equalization.py        # CLAHE implementation
│
└── bit_plane_slicing/
└── bit_plane.py                 # Bit-plane slicing implementation

Code

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/digital-image-processing.git
cd digital-image-processing
Install dependencies:

bash
pip install streamlit opencv-python matplotlib
🚀 Usage
Run the app with:

bash
streamlit run app.py
Then open the browser link (usually http://localhost:8501) to interact with the web interface.

🎨 Features
Upload colourful images (JPEG/PNG).

Apply GHE, LHE, or CLAHE on grayscale internally, with results displayed in RGB for clarity.

Perform bit-plane slicing and visualize all 8 planes.

Modern gradient background and styled headings for a dashboard-like feel.

📸 Example Workflow
Upload an image.

Select an operation from the dropdown:

Global Histogram Equalization (GHE)

Local Histogram Equalization (LHE)

CLAHE

Bit Plane Slicing

View the processed result inline.

🛠️ Technologies Used
Streamlit — for the interactive web UI

OpenCV — for image processing

Matplotlib — for bit-plane visualization

📌 Notes
Histogram equalization is applied on grayscale versions of the image, but results are converted back to RGB for display.

Bit-plane slicing always uses grayscale internally.

👨‍💻 Author
Yash Gupta

📄 License
This project is licensed under the MIT License.
