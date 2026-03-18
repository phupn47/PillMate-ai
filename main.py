import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = YOLO("pillcount.pt")

@app.websocket("/ws/pill-detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # รับข้อมูลภาพ Base64 จาก Frontend
            data = await websocket.receive_text()
            
            # แปลง Base64 เป็นภาพ OpenCV
            img_data = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                results = model(frame, conf=0.5)
                
                counts = {"round": 0, "oval": 0}
                for r in results:
                    for c in r.boxes.cls:
                        name = model.names[int(c)]
                        if "round" in name.lower(): counts["round"] += 1
                        elif "oval" in name.lower(): counts["oval"] += 1

                await websocket.send_json({
                    "round": counts["round"],
                    "oval": counts["oval"],
                    "total": counts["round"] + counts["oval"]
                })
    except Exception as e:
        print(f"Websocket Error: {e}")
    finally:
        await websocket.close()