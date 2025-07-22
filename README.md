# PneumonAI - Pneumonia Detection App

A deep learning application for pneumonia detection using VGG16 architecture.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/iZekCode/Pneumonia-Detection-VGG16.git
cd Pneumonia-Detection-VGG16
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. **Download the model file:**
   
   Due to file size limitations, the trained model (`pneumonAI_model.pth`) is not included in this repository. 
   
   You can:
   - Train your own model using the provided code
   - Or contact the repository owner for the pre-trained model file
   
   Place the model file in the root directory as `pneumonAI_model.pth`

4. Run the application:
```bash
python app.py
```

## Model Information

- Architecture: VGG16
- Task: Pneumonia Detection from X-ray images
- Model file: `pneumonAI_model.pth` (not included due to size constraints)

## File Structure

- `app.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `templates/` - HTML templates
- `assets/` - Static assets (images, documents)
- `pneumonAI_model.pth` - Trained model (to be downloaded separately)
