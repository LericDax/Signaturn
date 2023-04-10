import tkinter as tk
from tkinter import filedialog
import cv2
import os
import shutil
import numpy as np
import re
import pytesseract
from pytesseract import Output
from skimage.metrics import structural_similarity as compare_ssim

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  # Replace with the path to your Tesseract executable

def resize_image(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return thresh

def extract_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_signature_box(image):
    thresh = preprocess_image(image)
    contours = extract_contours(thresh)

    signature_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            signature_contour = contour

    if signature_contour is not None:
        x, y, w, h = cv2.boundingRect(signature_contour)
        signature_box = image[y:y+h, x:x+w]
        return signature_box
    else:
        return None

def compare_images(img1, img2):
    fixed_dim = (300, 300)
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize both img1_gray and img2_gray to fixed dimensions
    img1_gray = cv2.resize(img1_gray, fixed_dim)
    img2_gray = cv2.resize(img2_gray, fixed_dim)

    score, _ = compare_ssim(img1_gray, img2_gray, full=True)
    return score

def detect_name(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text.strip()

def group_similar_signatures(signatures, threshold=0.6):
    groups = []

    for signature, name in signatures:
        if signature is not None:
            added_to_existing_group = False
            for group in groups:
                if compare_images(group[0][0], signature) >= threshold:
                    group.append((signature, name))
                    added_to_existing_group = True
                    break

            if not added_to_existing_group:
                groups.append([(signature, name)])

    return groups

def save_signatures(groups, output_dir):
    for i, group in enumerate(groups):
        if group[0][1]:
            group_name = re.sub(r'\W+', '', group[0][1])
            group_name = group_name.replace(" ", "_")
            group_name = group_name[:100]  # Truncate the name to a maximum of 100 characters
        else:
            group_name = f"group_{i}"

        group_dir = os.path.join(output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        for j, (signature, name) in enumerate(group):  # Add an index variable using enumerate
            filename = f"signature_{j}.png"  # Replace group.index() with the index variable j
            cv2.imwrite(os.path.join(group_dir, filename), signature)



def start_processing(input_dir, output_dir):
    signatures = []
    for file in os.listdir(input_dir):
        image = cv2.imread(os.path.join(input_dir, file))
        
        # Detect signature box
        signature_box = detect_signature_box(image)

        # Detect name
        name = detect_name(image)

        # Save signature and name
        signatures.append((signature_box, name))
    
    # Group similar signatures
    groups = group_similar_signatures(signatures)
    
    # Save signatures into individual folders
    save_signatures(groups, output_dir)

# Create and configure the GUI
def select_folder():
    folder_path = filedialog.askdirectory()
    return folder_path

def on_process_click():
    input_dir = input_var.get()
    output_dir = output_var.get()
    start_processing(input_dir, output_dir)

root = tk.Tk()
root.title("Signature Grouping App")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

input_label = tk.Label(frame, text="Input Folder:")
input_label.grid(row=0, column=0, sticky="e")

input_var = tk.StringVar()
input_entry = tk.Entry(frame, textvariable=input_var)
input_entry.grid(row=0, column=1)

input_button = tk.Button(frame, text="Browse", command=lambda: input_var.set(select_folder()))
input_button.grid(row=0, column=2, padx=(5, 0))

output_label = tk.Label(frame, text="Output Folder:")
output_label.grid(row=1, column=0, sticky="e")

output_var = tk.StringVar()
output_entry = tk.Entry(frame, textvariable=output_var)
output_entry.grid(row=1, column=1)

output_button = tk.Button(frame, text="Browse", command=lambda: output_var.set(select_folder()))
output_button.grid(row=1, column=2, padx=(5, 0))

process_button = tk.Button(frame, text="Start Processing", command=on_process_click)
process_button.grid(row=2, column=1, pady=(10, 0))

root.mainloop()

