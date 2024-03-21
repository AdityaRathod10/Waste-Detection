import customtkinter as ctk
import matplotlib.pyplot as plt
from multiprocessing import Process, Value
import cv2
from tkinter import filedialog
from ultralytics import YOLO
import cvzone
import math
from PIL import Image, ImageTk
from collections import Counter
from tkinter import *
from PIL import Image
import time
import psycopg2
from tkinter import PhotoImage, Label
confidence_value = Value('d', 0.1)
box_thickness = 2
def start(stop_capture, confidence_value) :
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("C:\\Users\\Asus\\PycharmProjects\\Waste-Detection-Yolov8-project\\best weights\\best.pt")

    classNames = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    myColor = (0, 0, 255)

    last_added = {class_name: 0 for class_name in classNames}

    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS waste (
            id SERIAL PRIMARY KEY,
            class_name VARCHAR(255),
            box_xmin INTEGER,
            box_ymin INTEGER,
            box_xmax INTEGER,
            box_ymax INTEGER,
            confidence FLOAT
        )
    """)
    conn.commit()

    while True:
        if stop_capture.value:
            break
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam. Exiting ...")
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf > confidence_value.value:
                    if currentClass == 'BIODEGRADABLE':
                        myColor = (0, 255, 0)
                    elif currentClass == 'CARDBOARD':
                        myColor = (0, 0, 255)
                    elif currentClass == 'GLASS':
                        myColor = (255, 0, 0)
                    elif currentClass == 'METAL':
                        myColor = (255, 255, 0)
                    elif currentClass == 'PAPER':
                        myColor = (0, 255, 255)
                    elif currentClass == 'PLASTIC':
                        myColor = (255, 0, 255)

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                     (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(0, 0, 0), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)


                    if time.time() - last_added[currentClass] >= 7.0:
                        last_added[currentClass] = time.time()
                        cur.execute("INSERT INTO waste (class_name, box_xmin, box_ymin, box_xmax, box_ymax,confidence) VALUES (%s, %s, %s, %s, %s,%s)",
                                    (currentClass, x1, y1, x2, y2,confidence))


                        conn.commit()

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            visualize_data(graph_type)
            break

    conn.close()
def visualize_data(graph_type):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()
    cur.execute("SELECT class_name FROM waste;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    class_names = [row[0] for row in rows]

    class_counts = Counter(class_names)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    if graph_type == "bar":
        plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    elif graph_type == "pie":
        plt.pie(class_counts.values(), labels=class_counts.keys(), colors=colors, autopct='%1.1f%%')
    elif graph_type == "histogram":
        plt.hist(class_names, bins=len(class_counts), color=colors[0])
    elif graph_type == "line":
        plt.plot(list(class_counts.keys()), list(class_counts.values()), color=colors[0])

    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.title('Number of Detected Classes')
    plt.show()

def predict(confidence_value):
    classNames = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    color_dict = {"BIODEGRADABLE": (0,255,0), "CARDBOARD": (0,0,255), "GLASS": (255,0,0), "METAL": (255,255,0), "PAPER": (255,0,255), "PLASTIC": (0,255,255)}

    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if not filepath:
        return

    img = cv2.imread(filepath)
    img = cv2.resize(img, (1280, 720))

    model = YOLO("C:\\Users\\Asus\\PycharmProjects\\Waste-Detection-Yolov8-project\\best weights\\best.pt")
    names = model.names

    results = model(img, show=False)
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS waste (
            id SERIAL PRIMARY KEY,
            class_name VARCHAR(255),
            box_xmin INTEGER,
            box_ymin INTEGER,
            box_xmax INTEGER,
            box_ymax INTEGER,
            confidence FLOAT
        )
    """)

    conn.commit()

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for xyxy, c, confidence in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1 = (int(xyxy[0]),int(xyxy[1]))
            x2, y2 = (int(xyxy[2]),int(xyxy[3]))
            class_name = names[int(c)]
            conf = float(confidence)
            print(class_name)
            print(conf)
            print((x1,y1),(x2,y2))
            if conf > confidence_value:
                color = color_dict[class_name]
                cv2.rectangle(img,(x1, y1), (x2, y2), color, 2)
                text = f'{class_name}: {conf}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                if y1 < text_size[1]:
                    text_location = (x1, y1+text_size[1])
                else:
                    text_location = (x1, y1-10)
                cv2.putText(img, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                cur.execute("""
                    INSERT INTO waste (class_name, box_xmin, box_ymin, box_xmax, box_ymax,confidence )
                    VALUES (%s, %s, %s, %s, %s,%s)
                """, (class_name, x1, y1, x2, y2,conf))

            conn.commit()
    conn.close()

    cv2.imshow('Image', img)
    cv2.waitKey(0)

    return img