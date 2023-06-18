import os
import shutil
import librosa
from pathlib import Path
from scipy.io import wavfile
from moviepy.editor import VideoFileClip
from keras.utils import np_utils
import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

multimodal_model = keras.models.load_model('multimodal_model.h5')


# 데이터 전처리
def preprocess_video(video_path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cnn_data = []
    rnn_data = []

    cap = cv2.VideoCapture(video_path)
    count = 0
    while len(cnn_data) < 2:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                resized_img = cv2.resize(face_img, (224, 224))
                cnn_data.append(resized_img)
                count += 1
                if count >= 15:
                    break
        else:
            break

    if len(cnn_data) < 281:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile("audio.wav")
        y, sr = librosa.load("audio.wav", sr=44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc = mfcc[:, :400]
        rnn_data.append(mfcc)
        os.remove("audio.wav")

    cnn_data = np.array(cnn_data)
    rnn_data = np.array(rnn_data)
    return cnn_data, rnn_data

# 딥페이크 영상 유무 판별
def detect_deepfake(video_path):
    cnn_data, rnn_data = preprocess_video(video_path)

    cnn_data_np = np.array(cnn_data)
    rnn_data_np= np.array(rnn_data)

    def augment_data(data, target_size):
      # 증강된 데이터 배열 초기화
      augmented_data = np.empty((target_size,) + data.shape[1:])

      # RNN 데이터를 반전하여 복사
      for i in range(target_size):
          augmented_data[i] = np.flip(data[i % data.shape[0]], axis=0)

      return augmented_data

    # RNN 데이터 증강
    augmented_rnn_data = augment_data(rnn_data_np, cnn_data_np.shape[0])

    y_pred = multimodal_model.predict([cnn_data, augmented_rnn_data])

    #print(y_pred)
    max_prob = np.max(y_pred)
    print(max_prob)

    if max_prob < 0.5:
        return "Deepfake"
    else:
        return "Real"
# Streamlit 애플리케이션
def main():
    # 페이지 제목과 설명
    st.title("Deepfake Detection Demo")
    st.write("Upload a video file to detect if it contains deepfake content.")

    # 파일 업로드
    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file is not None:
        # 비디오 딥페이크 여부 검출
        st.write("Detecting deepfake...")
        preprocess_video(video_file)
