import av
from pathlib import Path
from pytubefix import YouTube
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer, VideoTransformerBase

import numpy as np
from io import BytesIO
import queue

import time
from collections import deque

import csv
import re
import requests
import datetime
import os
import io
import base64
import plotly.graph_objects as go

from .class_names import class_names

def styling_css():
    with open('./assets/css/general-style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
def create_fig(image, detected=False):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data_uri = base64.b64encode(buffer.getvalue()).decode()
    
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{image_data_uri}",
            x=0,
            y=image.size[1],
            xref="x",
            yref="y",
            sizex=image.size[0],
            sizey=image.size[1],
            layer="below"
        )
    )
    
    fig.update_layout(
        xaxis_range=[0, image.size[0]],
        yaxis_range=[0, image.size[1]],
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                showarrow=False,
                text="Detected Image" if detected else "Original Image",
                xref="paper",
                yref="paper"
            )
        ]
    )
    
    return fig

def convert_youtube_url(url):
    pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com/shorts/|youtube\.com/watch\?v=|youtu\.be/)([\w\-]{11})"
    match = re.search(pattern, url)
    
    if match:
        video_id = match.group(1)
        return f"https://youtu.be/{video_id}"
    return None


def _display_detected_frame(conf, model, youtube_url=""):
    if youtube_url:
        youtube_id = convert_youtube_url(youtube_url)
        if youtube_id:
            valid_url = youtube_id
            st.toast('Connecting', icon="🕒")
            
            try:
                yt = YouTube(valid_url)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                if not stream:
                    st.error("No suitable video stream found.")
                    return

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    video_path = tfile.name
                
                stream.download(output_path=os.path.dirname(video_path), filename=os.path.basename(video_path))
                
                results = model(source=video_path, stream=True, conf=conf, imgsz=640, save=True, device="cpu", vid_stride=1)
                displayed_dishes = set()

                detection_results = ""
                new_detections = False
                nutrition_data = []
                current_time = datetime.datetime.now()
                time_format = current_time.strftime("%d-%m-%Y")
                
                stop_button = st.button("Stop")
                stop_pressed = False

                st_frame = st.empty()

                frame_count = 0
                start_time = time.time()
                fps = 0


                st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
                # nutrition_placeholder = st.empty()

                for r in results:    
                    im_bgr = r.plot() 
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 1.0:
                        fps = frame_count / elapsed_time
                        start_time = time.time()
                        frame_count = 0
                    cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4) 

                    im_rgb = Image.fromarray(im_bgr[..., ::-1])  
                    im_rgb_resized = im_rgb.resize((640, 640))        
                    st_frame.image(im_rgb_resized, caption='Predicted Video', use_column_width=True)      
                    for pred in r.boxes: 
                        class_id = int(pred.cls[0].item())
                        class_name = class_names[int(class_id)]["name"]
                        confident = int(round(pred.conf[0].item(), 2)*100)


                        if isinstance(pred.xyxy, torch.Tensor):
                            boxes = pred.xyxy.cpu().numpy()
                        else:
                            boxes = pred.xyxy.numpy()
                    
                        image_np = r.orig_img 
                        
                        bounding_box_images = extract_bounding_box_image(image_np, boxes)

                        bbox_image_html = ""
                        if bounding_box_images:
                            bbox_image = bounding_box_images[0]
                            bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                            buffered = io.BytesIO()
                            bbox_image_pil.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'


                        if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                            detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"

                            displayed_dishes.add(class_name)
                            new_detections = True
                        elif class_name not in displayed_dishes:
                            displayed_dishes.add(class_name)
                            new_detections = True
                            
                            detection_results += (
                                f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
                                <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                                """)
                            
                            nutrition_data.append((
                                class_name,
                                confident
                            ))
                if new_detections:
                    scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
                    
                    st.markdown(scrollable_textbox, unsafe_allow_html=True)

                displayed_dishes.clear()


                # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir="/tmp") as csv_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                    csv_filename = csv_file.name
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Food Name", "Confidence (%)"])
                    writer.writerows(nutrition_data)
                with open(csv_filename, "rb") as file:
                    the_csv = file.read()  
                
                st.toast("Prediction completed. Results saved to CSV.", icon="✅")
                time.sleep(3000)
                download_csv = st.download_button(label="Download Predictions CSV",
                                data=the_csv,
                                file_name=f"{time_format}.csv", 
                                use_container_width=True,
                                key=f"download_csv3_button_{time_format}")
                if download_csv:
                    os.remove(csv_filename)
            except ConnectionError as e:
                st.error(f"Failed to open YouTube video stream: {e}")
        else:
            st.error("Invalid YouTube URL or unable to extract YouTube ID.")
    else:
        st.error("YouTube URL is required.")

@st.cache_resource
def load_model():
    # modelpath = Path(__file__).parent / "model/checkpoint_best_rpl.pth"
    modelpath = Path(__file__).parent / "model/vietfood67_yolov8s/best.pt"

    model = YOLO(modelpath, task='detect')
    return model

def resize_image(image):
    return image.resize((640, 640))



import torch

def extract_bounding_box_image(image, boxes):
    h, w = image.shape[:2]
    extracted_images = []

    for box in boxes:
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
 
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w, endX), min(h, endY)

        bbox_image = image[startY:endY, startX:endX]
        extracted_images.append(bbox_image)

    return extracted_images

def detect_image_result(detected_image, model):
    boxes = detected_image[0].boxes

    if boxes:
        detected_img_arr_RGB = detected_image[0].plot()[:, :, ::1]
        detected_img_arr_BGR = detected_image[0].plot()[:, :, ::-1]
        fig_detected = create_fig(detected_img_arr_BGR, detected=True)
        st.plotly_chart(fig_detected, use_container_width=True)

        current_time = datetime.datetime.now()
        time_format = current_time.strftime("%d-%m-%Y")

        # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir='/tmp') as img_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=tempfile.gettempdir()) as img_file:
            img_filename = img_file.name
            cv2.imwrite(img_filename, detected_img_arr_RGB)
        with open(img_filename, 'rb') as file:
            the_img = file.read()
        
            detection_results = ""
            count_results = ""
            count_dict = {}
            food_names = []
            nutrition_data = []
            confidences = []
            counts = []



            
            for r in detected_image[0]:
                for box in r.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = class_names[int(class_id)]["name"] 
                    food_names.append(class_name)
                    conf = int(round(box.conf[0].item(), 2)*100)
                    confidences.append(conf)


                    if isinstance(box.xyxy, torch.Tensor):
                        boxes = box.xyxy.cpu().numpy()
                    else:
                        boxes = box.xyxy.numpy()
                
                    image_np = r.orig_img 
                    
                    bounding_box_images = extract_bounding_box_image(image_np, boxes)

                    bbox_image_html = ""
                    if bounding_box_images:
                        bbox_image = bounding_box_images[0]
                        bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                        buffered = io.BytesIO()
                        bbox_image_pil.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'

                    

                    if class_id in count_dict:
                        count_dict[class_id] += 1
                    else:
                        count_dict[class_id] = 1

                    if class_name == "Con nguoi (Human)":
                        detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {conf}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"

                    else:
                        detection_results += (
                        f"""<p class="item-header">{count_dict[class_id]} ({conf}%): <b>{class_name}</b></p>
                        <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                        """)

                        nutrition_data.append((
                            class_name,
                            conf
                        ))
            # detection_results += total_nutrition_str
            # total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)

            for object_type, count in count_dict.items():
                the_name = class_names[object_type]["name"]
                counts.append(count)
                # detection_results += f"<b style='color: black;'>Count of {the_name}:</b> {count}<br>"
                count_results += f"""
                <p class="total-count-result-text">{the_name}: {count}<hr class="dash-line-below-count-results"></p>"""
                
            scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
            
            st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
            
            st.markdown(f'<div class="total-count-result-div">{count_results}</div>', unsafe_allow_html=True)
            st.markdown(scrollable_textbox, unsafe_allow_html=True)

            # rows = zip(food_names, confidences, counts)
            # rows = zip(nutrition_data)

            # with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir='/tmp') as csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                csv_filename = csv_file.name
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Food Name", "Confidence (%)"])
                writer.writerows(nutrition_data)
            with open(csv_filename, 'rb') as file:
                the_csv = file.read()
        col1, col2 = st.columns(2, gap="large")
        with col1:    
            download_pic = st.download_button(label="Download Predicted Image",
                                    data=the_img,
                                    mime="image/jpg",
                                    file_name=f"{time_format}.jpg", 
                                    use_container_width=True,
                                    key=f"download_pic_button_{time_format}")
            if download_pic:
                os.remove(img_filename)
        with col2: 
            download_csv = st.download_button(label="Download Predictions CSV", 
                               data=the_csv, 
                               file_name=f"{time_format}.csv", 
                               use_container_width=True,
                               key=f"download_csv_button_{time_format}")
            if download_csv:
                os.remove(csv_filename)
        st.divider()

    else:
        st.markdown("""<h5 class="total-count-result" id="no-food-detected">No food detected</h5>
                    <p class="result-nutri-container" id="no-food-descr">The model did not detect any foods in the uploaded image.  
            Please try with a different image or adjust the model's 
            confidence threshold and try again.</p>
                    """, unsafe_allow_html=True)  


def detect_image(conf, uploaded_file, model, url=False):
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
        
        if "is_reset" not in st.session_state:
            st.session_state.is_reset = False
        
        if "show_image" not in st.session_state:
            st.session_state.show_image = True

        reset_button = None
        predict_button = None
        
        def toggle_button(reset = False):
            st.session_state.button_clicked = not st.session_state.button_clicked
            st.session_state.show_image = not st.session_state.show_image
            if reset == True:
                st.session_state.is_reset = not st.session_state.is_reset
        
        original_image = st.empty()

        if url==False:
            uploaded_image = Image.open(uploaded_file)
        else:
            response = requests.get(uploaded_file)
            response.raise_for_status()
            uploaded_image = Image.open(BytesIO(response.content))

        resized_uploaded_image = resize_image(uploaded_image)

        if st.session_state.show_image and not st.session_state.is_reset and not st.session_state.button_clicked:   
            original_image = st.image(resized_uploaded_image, output_format="JPEG", use_column_width=True)

        if not st.session_state.is_reset:
            col1, col2 = st.columns([0.8, 0.2], gap="large")
            with col1:
                if st.session_state.show_image and not st.session_state.button_clicked and not original_image == st.empty():
                    st.markdown("**Original Image**")
                elif not st.session_state.show_image and st.session_state.button_clicked:
                    st.markdown("**Predicted Image**")
            with col2:
                if not st.session_state.button_clicked:
                    predict_button = st.button("Predict", use_container_width=True, type="primary", on_click=toggle_button)
                else:
                    reset_button = st.button("Reset", use_container_width=True, type="primary", on_click=toggle_button, args=[True])
                    uploaded_file = None
        if st.session_state.show_image and st.session_state.is_reset and not st.session_state.button_clicked:
            st.session_state.is_reset = False

        if st.session_state.button_clicked and not reset_button:
            with st.spinner("Running..."):
                detected_image = model.predict(resized_uploaded_image, conf=conf, imgsz=640)
                detect_image_result(detected_image, model)        

def detect_camera(conf, model, address):
    vid_cap = cv2.VideoCapture('rtsp://admin:' + address)
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  
    vid_cap.set(cv2.CAP_PROP_FPS, 15)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    
    while True:
        if vid_cap.isOpened():
            st.toast("Connected", icon="✅")
            break
        else:
            vid_cap.release()
            return
    try: 
        st_frame = st.empty()

        displayed_dishes = set()


        
        st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)

        frame_count = 0
        start_time = time.time()
        while True:   
            success, image = vid_cap.read()
            if success:
                # mirrored_frame = cv2.flip(image, 1)
                results = model.track(source=image, conf=conf, imgsz=640, save=False, device="cpu", stream=True)


                new_detections = False  
                detection_results = ""

                for r in results:
                    im_bgr = r.plot()
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 1.0:
                        fps = frame_count / elapsed_time
                        start_time = time.time()
                        frame_count = 0
                    cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                    im_rgb = Image.fromarray(im_bgr[..., ::-1])
                    st_frame.image(im_rgb, caption='Camera IP', use_column_width=True)

                    for pred in r.boxes:
                        class_id = int(pred.cls[0].item())
                        class_name = class_names[int(class_id)]["name"]
                        confident = int(round(pred.conf[0].item(), 2)*100)


                        if isinstance(pred.xyxy, torch.Tensor):
                            boxes = pred.xyxy.cpu().numpy()
                        else:
                            boxes = pred.xyxy.numpy()
                    
                        image_np = r.orig_img 
                        
                        bounding_box_images = extract_bounding_box_image(image_np, boxes)

                        bbox_image_html = ""
                        if bounding_box_images:
                            bbox_image = bounding_box_images[0]
                            bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                            buffered = io.BytesIO()
                            bbox_image_pil.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'


                        if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                            detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"
                            displayed_dishes.add(class_name)
                            new_detections = True
                        elif class_name not in displayed_dishes:
                            displayed_dishes.add(class_name)
                            new_detections = True

                            detection_results += (
                            f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
                            <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                            """)
                    
                        if new_detections:
                            scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
                            
                            st.markdown(scrollable_textbox, unsafe_allow_html=True)
            else:
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
    finally:
        vid_cap.release()
        displayed_dishes.clear()

from typing import List, NamedTuple
result_queue = queue.Queue(maxsize=12)

class Detection(NamedTuple):
    class_id: int
    class_name: str
    confident: float
    bbox_image_html: str

class VideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.prev_time = time.time()
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        mirrored_frame = cv2.flip(img, 1)
        results = self.model(source=mirrored_frame, conf=self.conf, imgsz=640, save=False, device="cpu", stream=True, vid_stride=80)
        # results = self.model.track(source=mirrored_frame, conf=self.conf, imgsz=640, save=False, device="cpu", stream=True)
        detections = []
        for r in results:
            im_bgr = r.plot()
            for pred in r.boxes:
                class_id = int(pred.cls[0].item())
                if isinstance(pred.xyxy, torch.Tensor):
                    boxes = pred.xyxy.cpu().numpy()
                else:
                    boxes = pred.xyxy.numpy()
            
                image_np = r.orig_img 
                
                bounding_box_images = extract_bounding_box_image(image_np, boxes)

                bbox_image_html = ""
                if bounding_box_images:
                    bbox_image = bounding_box_images[0]
                    bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                    buffered = io.BytesIO()
                    bbox_image_pil.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'
                detections.append(
                    Detection(
                        class_id = int(pred.cls[0].item()),
                        class_name = class_names[int(class_id)]["name"],
                        confident = int(round(pred.conf[0].item(), 2)*100),
                        bbox_image_html = bbox_image_html
                    )
                )

        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time

        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.prev_time = current_time
            self.frame_count = 0
        else:
            fps = self.frame_count / elapsed_time
  

        cv2.putText(im_rgb, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not result_queue.full():
            result_queue.put(detections)

        return av.VideoFrame.from_ndarray(im_rgb, format="rgb24")   



def detect_webcam(conf, model):
    webrtc_ctx = webrtc_streamer(
        key="webcam_1",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(conf, model),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        total_nutrition_placeholder = st.empty()
        st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
        results_placeholder = st.empty()
        
        while True:

            detections = result_queue.get()

            detection_results = ""
            new_detections = False
            displayed_dishes = set()

            for detection in detections:
                class_id = detection.class_id
                class_name = detection.class_name
                confident = detection.confident
                bbox_image_html = detection.bbox_image_html
                
                if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                    detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"

                    displayed_dishes.add(class_name)
                    new_detections = True
                elif class_name not in displayed_dishes:
                    displayed_dishes.add(class_name)
                    new_detections = True

                    detection_results += (
                    f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
                    <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                    """)

            results_placeholder.markdown(f"""<div class="result-nutri-container">{detection_results}</div>""", unsafe_allow_html=True)


    
            displayed_dishes.clear()
    
    result_queue.queue.clear()




def detect_video(conf, uploaded_file, model):
    if uploaded_file:
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
            temp_input_file.write(uploaded_file.read())
            temp_input_file_path = temp_input_file.name
        detect_from_file(conf=conf, video_file=temp_input_file_path, model=model)

def detect_from_file(conf, video_file, model):
    if video_file:
        cap = cv2.VideoCapture(video_file)

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%d-%m-%Y")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)

    # with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/tmp') as mp4_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=tempfile.gettempdir()) as mp4_file:
        mp4_filename = mp4_file.name
        out = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    with open(mp4_filename, "rb") as file:
        the_mp4 = file.read()

    st_frame = st.empty()

    

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        rewind_button = st.button("Rewind 10s", use_container_width=True)
    with col2:
        stop_button = st.button("Stop", use_container_width=True)
        stop_pressed = False
    with col3:
        fast_forward_button = st.button("Fast-forward 10s", use_container_width=True)

    frame_count = 0
    start_time = time.time()
        
    stop_pressed = False
    skip_frames = 0

    total_nutrition_placeholder = st.empty()
    st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
    

    displayed_dishes = set()
    detection_data = []
    while True:

        success, image = cap.read()

        if skip_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames)
            skip_frames = 0
        if rewind_button:
            skip_frames = -int(fps * 10) 
        if fast_forward_button:
            skip_frames = int(fps * 10)   
        if stop_button:
            stop_pressed = True

        if not success or stop_pressed:
            break

        results = model.predict(source=image, conf=conf, imgsz=640, save=False, device="cpu")

        new_detections = False  
        detection_results = ""

            
        for r in results:
            im_bgr = r.plot()           
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0
            cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            st_frame.image(im_rgb, caption='Predicted video', use_column_width=True)

    
            for pred in r.boxes:
                class_id = int(pred.cls[0].item())
                class_name = class_names[int(class_id)]["name"]
                confident = int(round(pred.conf[0].item(), 2)*100)
                confident = int(round(pred.conf[0].item(), 2)*100)

                if isinstance(pred.xyxy, torch.Tensor):
                    boxes = pred.xyxy.cpu().numpy()
                else:
                    boxes = pred.xyxy.numpy()
            
                image_np = r.orig_img 
                
                bounding_box_images = extract_bounding_box_image(image_np, boxes)

                bbox_image_html = ""
                if bounding_box_images:
                    bbox_image = bounding_box_images[0]
                    bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                    buffered = io.BytesIO()
                    bbox_image_pil.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'


                if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                    detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"
                    displayed_dishes.add(class_name)
                    new_detections = True
                elif class_name not in displayed_dishes:
                    displayed_dishes.add(class_name)
                    new_detections = True

                    detection_results += (
                    f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
                    <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                    """)
                    
                    detection_data.append((
                        class_name,
                        confident
                    ))
                
            if new_detections:
                scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
                
                st.markdown(scrollable_textbox, unsafe_allow_html=True)




            if stop_button:
                stop_pressed = True
                stop_button = None
                break

        cap.release()
        out.release()
        displayed_dishes.clear()


        # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir="/tmp") as csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
            csv_filename = csv_file.name
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Food Name", "Confidence (%)"])
            writer.writerows(detection_data)
        with open(csv_filename, "rb") as file:
            the_csv = file.read()
        
        col1, col2 = st.columns(2, gap="large")
        with col1:    
            download_video = st.download_button(label="Download Processed Video",
                                    data=the_mp4,
                                    mime="video/mp4",
                                    file_name=f"{timestamp}.mp4", 
                                    use_container_width=True,)
            if download_video:
                os.remove(mp4_filename)
        with col2: 
            download_csv = st.download_button(label="Download Predictions CSV", 
                                data=the_csv, 
                                file_name=f"{timestamp}.csv", 
                                use_container_width=True,)
            if download_csv:
                os.remove(csv_filename)