import streamlit as st

import base64 
from pathlib import Path

from UI.utils import detect_image, detect_video, detect_webcam, load_model

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:"
    
)



st.markdown('''
    <div id="top-section"></div>
    ''', unsafe_allow_html=True)
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Convert your image to base64
img_path = Path(__file__).parent / 'UI/assets/img/bg-about-cuisine.png'
img_base64 = img_to_base64(img_path)

import os


def get_available_models():
    model_dir = Path(__file__).parent / "UI/model"
    # Find all .pt and .pth files recursively
    models = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith((".pt", ".pth")):
                full_path = Path(root) / file
                # Create a relative path for display or just use filename if unique
                # Let's use relative path from UI/model for clarity if nested
                try:
                    rel_path = full_path.relative_to(model_dir)
                    
                    # Generate a nicer display name
                    if file == 'best.pt':
                        # Use parent folder name for YOLO models named best.pt
                        parent_name = full_path.parent.name
                        if "yolov8n" in parent_name:
                             display_name = "YOLOv8n"
                        elif "yolov8s" in parent_name:
                             display_name = "YOLOv8s"
                        else:
                             display_name = parent_name
                    elif file == 'checkpoint_best_rpl.pth':
                        display_name = "Faster R-CNN"
                    else:
                        # Fallback: filename without extension, formatted
                        display_name = file.rsplit('.', 1)[0].replace('_', ' ').title()
                        
                    models.append((display_name, str(full_path)))
                except ValueError:
                    models.append((file, str(full_path)))
    return models

def render_content():
    st.markdown(f'''<br><br>''', unsafe_allow_html=True)
    
    # Model Selection
    available_models = get_available_models()
    if not available_models:
        st.error("No models found in UI/model directory!")
        return

    # Default to the first one or a specific one if needed
    model_options = [m[0] for m in available_models]
    selected_model_name = st.selectbox("Select Model", model_options)
    
    selected_model_path = next(m[1] for m in available_models if m[0] == selected_model_name)
    
    model1 = load_model(selected_model_path)

    st.markdown("""
    <style>
    /* Style the tab labels */
    button[data-baseweb="tab"] {
        padding: calc(8px + 0.2vw) calc(8px + 0.5vw);
        gap: 0;

    }
    button[data-baseweb="tab"] p {
        font-size: calc(9px + 0.3vw) !important;
        font-weight: 500 !important;        
    }
    
    div[data-baseweb="tab-list"] {
        gap: 0;
    }
    /* Style the active tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--button-color-yellow); /* Active tab color */
        border-radius: 8px 7px 0 0;
        color: black;
    }

    /* Style the inactive tabs */
    button[data-baseweb="tab"][aria-selected="false"] {
        color: var(--grey-code-expander);
    }
    
    div[data-baseweb="tab-border"] {
    }
    </style>
""", unsafe_allow_html=True)
        
    tab1, tab2, tab3 = st.tabs(["Image", "Video", "Webcam"])

    with tab1:
        st.subheader("Image Upload :frame_with_picture:")


        st.markdown(f'''
<style>
[data-testid="stExpanderDetails"] ul li {{
    font-size: calc(12px + 0.1vw);
    margin: 1rem 0 1rem 1.5rem;
    color: black
}}
.stExpander p {{
    font-size: calc(13px + 0.1vw);
    font-weight: 700;
    color: var(--brown);
    padding-left: 0.5rem;
}}
.st-emotion-cache-1h9usn1 {{
    background-color: var(--button-color-yellor);
    font-size: calc(16px +1vw);
}}

[data-testid="stExpanderDetails"] {{
    background-color: var(--grey-light);
    border-radius: 8px;
}}
</style>
                    ''', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a picture", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            detect_image(0.5, model=model1, uploaded_file=uploaded_file)

    with tab2:
                    
        st.subheader("Video Upload :movie_camera:")

        
        uploaded_clip = st.file_uploader("Choose a clip", accept_multiple_files=False, type=['mp4'])
        if uploaded_clip:
            detect_video(conf=0.5, uploaded_file=uploaded_clip, model=model1)



    with tab3:
        
        st.header("Webcam :camera:")

        detect_webcam(0.5, model=model1)



    st.markdown('''
    <div>
        <a href="#top-section" class="top-button" onclick="smoothScroll(event, 'top-section')">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" width="16" height="16">
            <path d="M240.971 130.524l194.343 194.343c9.373 9.373 9.373 24.569 0 33.941l-22.667 22.667c-9.357 9.357-24.522 9.375-33.901.04L224 227.495 69.255 381.516c-9.379 9.335-24.544 9.317-33.901-.04l-22.667-22.667c-9.373-9.373-9.373-24.569 0-33.941L207.03 130.525c9.372-9.373 24.568-9.373 33.941-.001z"/>
        </svg>
        </a>                
    </div>
    
    <script>
    function smoothScroll(event, targetId) {
        event.preventDefault();
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    }
    </script>
                ''', unsafe_allow_html=True)

# Nav bar
def navbar(active_page):
    return f"""
    <div class="custom-navbar">
        <div class="nav-items">
            <a href="./" target="_self" class="nav-item {'active' if active_page == 'Home' else ''}">🏠 Home</a>
        </div>
        <a href="https://github.com/Anbt0106" target="_blank" class="nav-item">
            <svg id="github-icon" height="32" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="32" data-view-component="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" fill="currentColor"></path>
            </svg>
        </a>
    </div>
    """

def styling_css():
    css_path = Path(__file__).parent / 'UI/assets/css/general-style.css'
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
        

    

# Main app logic
def main():
        # Get the current page from the URL
    styling_css()
    query_params = st.query_params
    path = query_params.get("page", ["home"])[0].lower()
    
    # Always render the navbar
    st.markdown(navbar('Home' if path == 'home' else 'About'), unsafe_allow_html=True)
    
    if path == "about":
        st.markdown('<h1 style="color: white; font-size: 40px;">About Section</h1>', unsafe_allow_html=True)
        st.write("This is the About section. Here you can add information about your project or organization.")
    else:
        render_content()

if __name__ == "__main__":
    main()
