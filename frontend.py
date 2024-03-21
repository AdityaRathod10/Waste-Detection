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

from backend import *
def create_child_window():
    stop_capture = Value('b', False)

    child_window = ctk.CTkToplevel(root)
    child_window.geometry("800x600")
    frame = ctk.CTkFrame(child_window,width= 700, height= 500)
    frame.propagate(False)
    frame.pack()
    start_button = ctk.CTkButton(frame, text="Start", command=lambda: Process(target=start, args=(stop_capture, confidence_value)).start(),font= ("Arial",20), hover_color= "darkblue")
    start_button.pack()
    predict_button = ctk.CTkButton(frame, text="Predict Image", command=lambda: predict(confidence_value.value),font=("Arial",20),hover_color= "darkblue")
    predict_button.pack()
    def close():
        child_window.destroy()
        child_window.update()
    close_but = ctk.CTkButton(frame, text = "Close", command = close,hover_color= "darkblue", corner_radius=50)
    close_but.pack()
    graph_type = StringVar()
    radiobutton1= ctk.CTkRadioButton(frame, text="Bar Graph", variable=graph_type, value="bar", font = ("Arial",20))
    radiobutton2= ctk.CTkRadioButton(frame, text="Pie Chart", variable=graph_type, value="pie", font = ("Arial",20))
    radiobutton3= ctk.CTkRadioButton(frame, text="Histogram", variable=graph_type, value="histogram", font = ("Arial",20))
    radiobutton4= ctk.CTkRadioButton(frame, text="Line Graph", variable=graph_type, value="line", font = ("Arial",20))
    radiobutton1.pack(padx=20, pady=10)
    radiobutton2.pack(padx=20, pady=10)
    radiobutton3.pack(padx=20, pady=10)
    radiobutton4.pack(padx=20, pady=10)
    visualize_button = ctk.CTkButton(frame, text="Visualize Your Data", command=lambda: visualize_data(graph_type.get()),font=("Arial",20),hover_color= "darkblue")
    visualize_button.pack(padx =20, pady = 10)
    def sliding(value):
        label.configure(text = value)
        confidence_value.value = float(value)
        child_window.update()
    slider = ctk.CTkSlider(frame, from_=0, to= 1, command = sliding)
    slider.pack()
    slider.set(0)
    label = ctk.CTkLabel(frame, text = slider.get(), font =("Arial",20))
    label.pack()

if __name__ == "__main__":
    root = ctk.CTk()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    root.title("Waste Classifier")
    root.geometry("1980x1080")
    frame = ctk.CTkFrame(root, width=1400, height=900)
    frame.pack_propagate(False)
    frame.pack()
    mode = "dark"

    def change():

        global mode
        if mode == "dark":
            ctk.set_appearance_mode("light")
            mode = "light"

        else:
            ctk.set_appearance_mode("dark")
            mode = "dark"


    Change = ctk.CTkButton(frame, text=".", command=change,height = 20,width =20,hover_color= "darkblue")
    Change.pack(padx = 100, pady = 20)

    bg_image = Image.open("C:\\Users\\Asus\\PycharmProjects\\Waste-Detection-Yolov8-project\\Designer.jpg")
    bg_photo = ImageTk.PhotoImage(bg_image)

    bg_canvas = Canvas(frame, width=800, height=350)
    bg_canvas.pack(fill="both", expand=True)

    bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    title = ctk.CTkLabel(frame, text="Waste Classifier", font=("Arial Black", 47))
    title.pack()

    tips = [
        "Tip 1: Embark on your journey towards a cleaner world by clicking on the ‘Start’ button.",
    ]
    for tip in tips:
        tip_label = ctk.CTkLabel(frame, text=tip, font=("Arial Black", 16))
        tip_label.pack()

    get_started_button = ctk.CTkButton(frame, text="Get Started", command=create_child_window, font = ("Arial",20),hover_color= "darkblue",border_width = 10,border_color= "white")
    get_started_button.pack()

    root.mainloop()
