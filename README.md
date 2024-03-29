Signaturn

© 2023 LAK, Leric Dax, and Azoth Corp.
https://github.com/LericDax/Signaturn
www.LericDax.com

Signaturn is a Python application that groups and organizes scanned documents with signatures. 
The application uses OpenCV, Tesseract OCR, and Tkinter to process images and group them based on the similarity of the signatures. 
Signaturn also extracts the name from the document and uses it as a folder name for better organization.



Installation

Clone the repository or download the source code.

Create a virtual environment (or at least a terminal!) and activate it.



Install the required packages using the following command:


	pip install -r requirements.txt
	
Install Tesseract OCR:
	For Windows, download the Tesseract installer from here (https://github.com/UB-Mannheim/tesseract/wiki). 

After installation, set the path to the Tesseract executable by updating the following line in the script:

	pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
	
Run the application using the following command:

	python main.py
	
Click "Browse" next to "Input Folder" and select the folder containing the scanned images (JPEG, JPG, or PNG) with signatures.

Click "Browse" next to "Output Folder" and select the folder where you want to save the grouped signatures.

Click the "Start Processing" button to begin grouping the signatures. The images will be saved in subfolders within the specified output folder, named after the detected names or as "group_n" if the name is not detected.




Dependencies
tkinter
opencv-python
numpy
pytesseract
scikit-image
