import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

from image_editor_backend import ImageEditor

def show_masks(image, anns):
    if len(anns) == 0:
        return image

    sorted_anns = sorted(enumerate(anns), key=(lambda x: x[1]['area']), reverse=True)
    img = np.array(image)
    
    for original_idx, ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        img[m] = img[m] * 0.5 + np.array(color_mask) * 255 * 0.5

        # Find contours and compute centroid
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img, str(original_idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return Image.fromarray(img.astype(np.uint8))

st.title("AI Image Editor")

editor = ImageEditor()
editor.load_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Masks"):
        processed_image = editor.load_and_process_image(uploaded_file)
        masks = editor.generate_masks()

        masked_image = show_masks(processed_image, masks)
        st.image(masked_image, caption="Segmented Image", use_column_width=True)

        mask_id = st.number_input("Select a mask ID", min_value=0, max_value=len(masks)-1, value=0, step=1)
        prompt = st.text_input("Enter a prompt for image generation")

        if st.button("Generate New Image"):
            with st.spinner("Generating new image..."):
                result = editor.inpaint_image(mask_id, prompt)
                st.image(result, caption="Generated Image", use_column_width=True)

                # Save button
                buf = BytesIO()
                result.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png"
                )