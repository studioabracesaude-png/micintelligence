from fastapi import FastAPI, UploadFile, File
import mediapipe as mp
import cv2
import numpy as np
import tempfile

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.post("/avaliar")
async def avaliar_video(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)

    joelho_valgo_count = 0
    profundidade_ok = 0
    frames_analisados = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frames_analisados += 1

            quadril = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            joelho = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            tornozelo = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

            if joelho.x < tornozelo.x:
                joelho_valgo_count += 1

            if quadril.y > joelho.y:
                profundidade_ok += 1

    cap.release()

    score_valgo = 100 - (joelho_valgo_count / max(frames_analisados,1)) * 100
    score_profundidade = (profundidade_ok / max(frames_analisados,1)) * 100

    score_final = (score_valgo + score_profundidade) / 2

    return {
        "score_final": round(score_final, 2),
        "valgo_score": round(score_valgo, 2),
        "profundidade_score": round(score_profundidade, 2)
    }