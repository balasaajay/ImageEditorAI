# ImageEditorAI

### To run the app:

1. Install deps:

    ```bash
    pip install streamlit numpy torch torchvision Pillow opencv-python segment-anything diffusers transformers matplotlib
    ```

2. Download SAM checkpoints:
    
    ```bash
    cd app
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_chkpts/sam_vit_h_4b8939.pth
    ```

3. Run the Streamlit app:
    ```
    streamlit run app.py
    ```