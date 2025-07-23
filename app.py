import streamlit as st
import numpy as np
import cv2
from PIL import Image
from stl import mesh
from skimage import exposure
import tempfile
import os

def image_to_lithophane(image, max_thickness=4.0, min_thickness=0.8, scale=0.5):
    # Convert image to grayscale and normalize
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(
        img_gray,
        (int(img_gray.shape[1]*scale), int(img_gray.shape[0]*scale))
    )
    img_eq = exposure.rescale_intensity(img_resized, out_range=(0, 1))
    # Height map: white = thinnest, black = thickest
    heights = min_thickness + (1 - img_eq) * (max_thickness - min_thickness)
    height, width = heights.shape
    vertices = []
    for y in range(height):
        for x in range(width):
            vertices.append([x, y, heights[y, x]])
    vertices = np.array(vertices, dtype=np.float32)
    faces = []
    def get_idx(x, y): return y * width + x
    for y in range(height-1):
        for x in range(width-1):
            idx0, idx1, idx2, idx3 = get_idx(x, y), get_idx(x+1, y), get_idx(x, y+1), get_idx(x+1, y+1)
            faces.append([idx0, idx1, idx2])
            faces.append([idx1, idx3, idx2])
    faces = np.array(faces)
    litho_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            litho_mesh.vectors[i][j] = vertices[f[j], :]
    return litho_mesh

st.title('Image to Lithophane STL Converter By Santosh Dharamsale')

uploaded_file = st.file_uploader("Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
max_thick = st.slider("Maximum Thickness (mm)", 2.0, 8.0, 4.0)
min_thick = st.slider("Minimum Thickness (mm)", 0.2, 2.0, 0.8)
scale_factor = st.slider("Scale (for faster processing, lower = smaller image)", 0.1, 1.0, 0.5)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    width, height = image.size
    st.write(f"**Original Image size:** {width} x {height} pixels")

    # Aspect Ratio Lock Checkbox
    aspect_lock = st.checkbox("Lock Aspect Ratio", value=True)

    # Width adjustment
    new_width = st.slider("New Width (px)", 32, width, width)
    if aspect_lock:
        aspect_ratio = height / width
        new_height = int(new_width * aspect_ratio)
        st.write(f"New Height (auto): {new_height} pixels")
    else:
        new_height = st.slider("New Height (px)", 32, height, height)

    # Resize for preview/use
    resized_image = image.resize((new_width, new_height))
    st.image(resized_image, caption="Resized Image", use_container_width=True)
    st.write(f"**Resized Image size:** {new_width} x {new_height} pixels")

    # Use resized image for further processing
    if st.button("Generate Lithophane STL"):
        with st.spinner("Generating STL..."):
            litho_mesh = image_to_lithophane(resized_image, max_thick, min_thick, scale_factor)
            tmpdir = tempfile.gettempdir()
            stl_path = os.path.join(tmpdir, "lithophane.stl")
            litho_mesh.save(stl_path)
        with open(stl_path, "rb") as f:
            st.download_button(
                label="Download STL File",
                data=f,
                file_name="lithophane.stl",
                mime="application/sla"
            )
        st.success("STL file is ready for download.")

st.markdown("""
- Adjust thickness, scale, and image size to suit your 3D printing preferences.
- For best results, upload clear, high-contrast images.
""")
