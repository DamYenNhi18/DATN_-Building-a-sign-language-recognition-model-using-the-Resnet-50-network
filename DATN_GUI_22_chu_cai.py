import customtkinter as ctk
import tkinter as tk
from tkinter import Menu, messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame
import math
import threading
import time
from collections import deque
from tensorflow.keras.applications.resnet50 import preprocess_input

model = tf.keras.models.load_model(r"D:\DoAnTotNghiep\model_trainfull_ImageNet_22_chu_cai\best_model_00001_3 (1).keras")
#label_dict = {'D': 0, 'H': 1, 'U': 2, 'V': 3, 'nothing' : 4}
#reversed_dict = {value: key for key, value in label_dict.items()}

# Load model
#model = tf.keras.models.load_model(r"D:\DoAnTotNghiep\LearningRate10e-5\best_model_00001_2.keras")
label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
              'O': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'U': 18, 'V': 19, 'X': 20, 'Y': 21}
reversed_dict = {v: k for k, v in label_dict.items()}

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Khởi tạo pygame
pygame.mixer.init()

# Hàm để phát âm giọng nói 
def speak_cached(letter):
    if not sound_enabled:
        return
    filename = f"voices/{letter}.mp3"
    def play():
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    threading.Thread(target=play).start()


""" Thiết lập giao diện"""
# Tạo cửa sổ chính với CustomTkinter
window = ctk.CTk()
window.title("Mô hình Nhận diện Thủ ngữ")
window.geometry("1200x800")
window.configure(fg_color="#ecf0f1")


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
# ==== HEADER ====
header = ctk.CTkFrame(window, fg_color="#2c3e50", height= 120)
header.pack(side="top", fill="x")

# Tải ảnh logo PNG
logo_image = Image.open(r"C:\Users\ASUS\Downloads\logo-bach-khoa-dongphucsongphu2.png")
logo_ctk_image = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(100, 100))

# ==== Logo bên trái ====
logo_label = ctk.CTkLabel(header, image=logo_ctk_image, text="")
logo_label.place(x=110, y=10)  # Cách trái 20px, cách trên 30px cho đẹp

# ==== Khung chứa tiêu đề và slogan (Frame nhỏ giữa header) ====
title_frame = ctk.CTkFrame(header, fg_color="transparent")  # transparent = trong suốt
#title_frame.place(relx=0.5, rely=0.5, anchor="center")  # Căn giữa header



# ==== Tiêu đề nhỏ ====
slogan_label = ctk.CTkLabel(title_frame, text="Đồ án Tốt nghiệp",
                            font=("Roboto", 26), text_color="white")
slogan_label.pack(pady = 5)

# ==== Tiêu đề chính ====
title_label = ctk.CTkLabel(title_frame, text="Đề tài: MÔ HÌNH NHẬN DẠNG THỦ NGỮ",
                           font=("Arial", 26, "bold"), text_color="white")
title_label.pack(pady = 5)


# ==== MAIN AREA ====
main_frame = ctk.CTkFrame(window, fg_color="#f7f9fb")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Chia grid thành 2 cột
main_frame.grid_columnconfigure(0, weight=10)  # Sidebar 10%
main_frame.grid_columnconfigure(1, weight=50)   # Nội dung chính 50%
main_frame.grid_rowconfigure(0, weight=1)      # Một dòng duy nhất, full height



# ======= SIDEBAR =======   Khu vực chứa các nút chức năng chínhchính
sidebar = ctk.CTkFrame(main_frame, fg_color="#ecf0f1", corner_radius=10)
sidebar.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)  # sticky="nsew" để fill đủ chiều

button_font = ("Roboto", 16)
button_color = "#3498db"
# Các nút bên trái: Mở/Tắt camera → chọn chế độ → Bật/Tắt âm thanh → Bắt đầu/Dừng → Khởi động lại → ThoátThoát
btn_toggle_camera = ctk.CTkButton(sidebar, text="📹 Mở Camera", font=button_font, fg_color= button_color, height=45, corner_radius=8, command=lambda: toggle_camera())
btn_toggle_camera.pack(pady=5, padx=8, fill="x")

btn_option = ctk.CTkButton(sidebar, text="🔠 Chế độ nhận dạng", font=button_font, fg_color= button_color, height=45, corner_radius=8, command=lambda: option_detect())
btn_option.pack(pady=5, padx=8, fill="x")

btn_toggle_sound = ctk.CTkButton(sidebar, text="🔊 Bật âm thanh",font=button_font, fg_color= button_color, height=45, corner_radius=8, command=lambda: toggle_sound())
btn_toggle_sound.pack(pady=5, padx=8, fill="x")

btn_toggle_recognize = ctk.CTkButton(sidebar, text="▶ Start", font=button_font, fg_color="#27ae60", hover_color="#219150", height=45, corner_radius=8, command=lambda: toggle_tracking())
btn_toggle_recognize.pack(pady=5, padx=8, fill="x")

btn_reset = ctk.CTkButton(sidebar, text="🔄 Reset", font=button_font, fg_color="#2c3e50", hover_color="#34495e", height=45, corner_radius=8, command=lambda: reset())
btn_reset.pack(pady=5, padx=8, fill="x")

exit_button = ctk.CTkButton(sidebar, text="❌ Thoát", font=button_font, fg_color="#e74c3c", hover_color="#c0392b", height=45, corner_radius=8, command=lambda: exit_window())
exit_button.pack(pady=5, padx=8, fill="x")

# ==== Phần hiển thị camera và kết quả ====
center_frame = ctk.CTkFrame(main_frame, fg_color="white", corner_radius=10)
center_frame.grid(row=0, column=1, sticky="nsew", padx=(10,10), pady=5)

camera_frame = ctk.CTkFrame(center_frame,width=700, height=500, border_width=3, border_color="#3498db", corner_radius=10)
camera_frame.pack(pady=20, padx=20)

gray_background = np.full((480, 640, 3), (236, 240, 241), dtype=np.uint8)
empty_image_pil = Image.fromarray(gray_background)

# Chuyển sang `CTkImage`
empty_image_ctk = ctk.CTkImage(light_image=empty_image_pil, size=(640, 480))

# Hiển thị video trên giao diện
video_label = ctk.CTkLabel(master=camera_frame, image=empty_image_ctk, text="", width=640, height=480)
video_label.image = empty_image_ctk  # Giữ tham chiếu để ảnh không bị xóa
video_label.pack()



# Nhãn trạng thái
status_label = ctk.CTkLabel(center_frame, text="", font=("Segoe UI", 20), text_color="#2c3e50")
status_label.pack(pady=5)

label_result = ctk.CTkLabel(center_frame, text="Kết quả:",font=("Roboto",25,"bold"), anchor="w", text_color="#2c3e50")
label_result.pack(padx=20, anchor="w")
#label_result.pack_forget()

recognized_text = tk.StringVar(value="")
single_char_text = tk.StringVar()

#esult_textbox = ctk.CTkLabel(center_frame, textvariable=recognized_text, font=("Roboto",25), fg_color="#ecf0f1", text_color="black", corner_radius=8, anchor="w")
result_textbox = ctk.CTkLabel(center_frame, text="", font=("Roboto",25), fg_color="#ecf0f1", text_color="black", corner_radius=8, anchor="center")

result_textbox.pack(padx=20, pady=(0, 10), fill="x")

# ==== BUTTONS UNDER RESULT ====
buttons_frame = ctk.CTkFrame(center_frame, fg_color="white")
buttons_frame.pack(pady=5, padx=0, anchor="center")

btn_clear_all = ctk.CTkButton(buttons_frame, text="🗑 Xóa toàn bộ", font=button_font, fg_color= "#ecf0f1", text_color="black", command= lambda: clear_all_text())
#btn_clear_all.pack(side="left", padx=10, fill = "x")
btn_clear_all.pack_forget()

btn_space = ctk.CTkButton(buttons_frame, text="␣ Space", font=button_font,fg_color= "#ecf0f1", text_color="black",  command=lambda: add_space())
#btn_space.pack(side="left", padx=10, fill = "x")
btn_space.pack_forget()

btn_delete_one = ctk.CTkButton(buttons_frame, text="⌫ Xóa ", font=button_font, fg_color= "#ecf0f1", text_color="black", command= lambda: delete_last_character())
#btn_delete_one.pack(side="left", padx=10, fill = "x")
btn_delete_one.pack_forget()

# ==== FOOTER ====
footer = ctk.CTkLabel(center_frame, text="© 2025 | Version 1.0 · Dam Yen Nhi", font=("Roboto",13), text_color="#7f8c8d")
footer.place(relx=1, rely=1, anchor="se")  # 'sw' là "south-west" = góc dưới bên trái

def align_title_to_camera():
    window.update_idletasks()

    # Lấy vị trí khung camera
    camera_x = camera_frame.winfo_rootx()
    camera_width = camera_frame.winfo_width()
    camera_center = camera_x + camera_width // 2

    # Lấy vị trí gốc của window
    window_x = window.winfo_rootx()

    # Tính toạ độ x của tiêu đề trong cửa sổ
    new_title_x = camera_center - window_x

    # Cập nhật vị trí title_frame cho căn giữa với camera
    title_frame.place(x=new_title_x, rely=0.5, anchor="center")

# Gọi sau khi giao diện render
window.after(200, align_title_to_camera)


""" Khai báo biến """

last_detected_time = time.time() # Đếm thời gian không phát hiện kí tự tay
hand_was_present = True 
has_detected = False  # Đã từng nhận diện được chữ cái hay chưa


cap = None
is_camera_on = False
is_recognizing = False
tracking_active = False
imgSize = 224
frame_for_prediction = None
prediction_buffer = deque(maxlen=3)
last_spoken_char = ""
last_speak_time = 0
sound_enabled = False  # Bật âm thanh mặc định
speak_delay = 2
space_added = False
recognition_mode = None  # Biến chọn chế độ
tracking_option = False





""" Hàm xử lý ảnh đầu vào từ webcam"""
def preprocess_image(imCrop):
    imWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    h_crop, w_crop = imCrop.shape[:2]
    aspectRatio = h_crop / w_crop

    if aspectRatio > 1:
        k = imgSize / h_crop
        wCal = math.ceil(k * w_crop)
        imgResize = cv2.resize(imCrop, (wCal, imgSize))
        w_gap = (imgSize - wCal) // 2
        imWhite[:, w_gap:w_gap + wCal] = imgResize
    else:
        k = imgSize / w_crop
        hCal = math.ceil(k * w_crop)
        imgResize = cv2.resize(imCrop, (imgSize, hCal))
        h_gap = (imgSize - hCal) // 2
        imWhite[h_gap:h_gap + hCal, :] = imgResize

    img_input = cv2.cvtColor(imWhite, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_input, (224, 224))
    #img_input = img_input.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    img_input = preprocess_input(img_input)
    return img_input
    

""" Hàm nhận diện """
def prediction_worker():
    global frame_for_prediction, prediction_buffer, last_spoken_char, last_speak_time
    global last_detected_time, recognized_text, has_detected
    while True:
        if frame_for_prediction is not None:
            try:
                img_input = preprocess_image(frame_for_prediction)
                prediction = model.predict(img_input)
                confidence = np.max(prediction)
                predicted_label = np.argmax(prediction)
                print(predicted_label)
                print(confidence)

                if confidence >= 0.9:
                    predicted_char1 = reversed_dict[predicted_label]
                    if predicted_char1 == "nothing":
                        predicted_char = ""
                    else:
                         predicted_char = predicted_char1
                    prediction_buffer.append(predicted_char)
                    has_detected = True 

                    if len(prediction_buffer) == prediction_buffer.maxlen and all(c == predicted_char for c in prediction_buffer):
                        if recognition_mode == "sequence": # chọn chế độ hiển thị chuỗi
                            if time.time() - last_detected_time > 1.5:
                                recognized_text.set(recognized_text.get() + predicted_char)
                                last_detected_time = time.time()
                               
                        if recognition_mode == "letter":
                            single_char_text.set(f"Chữ {predicted_char}")        
                            current_time = time.time()
                            #if predicted_char != last_spoken_char and current_time - last_speak_time >= speak_delay:
                            if current_time - last_speak_time >= speak_delay:    
                                speak_cached(predicted_char)
                                last_spoken_char = predicted_char
                                last_speak_time = current_time
                                    
                            #update_result(f"Chữ: {predicted_char}", "#ecf0f1")
                            

                        
                else:
                    prediction_buffer.clear()
                    single_char_text.set("")
                    #update_result("", "Không phải chữ cái", "red")

            except Exception as e:
                print("Lỗi trong luồng dự đoán:", e)
            frame_for_prediction = None
        time.sleep(0.05)

""" Hàm xử lý khung hình từ webcam """  
def update_frame():
    global frame_for_prediction
    global last_detected_time
    global space_added, recognized_text, has_detected
    if not is_camera_on:
        return

    ret, frame = cap.read()
    if not ret:
        return
    
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    #if is_recognizing and results.multi_hand_landmarks:
    if tracking_active:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                if y_max - y_min > 0 and x_max - x_min > 0:
                    imCrop = frame[y_min-20:y_max+20, x_min-20:x_max+20]
                    cv2.rectangle(display_frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)
                    frame_for_prediction = imCrop.copy()
        else:
            if recognition_mode == "letter":
                single_char_text.set("")


    img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    # Chuyển đổi ảnh từ NumPy sang PIL
    img_pil = Image.fromarray(img_rgb)
    img_ctk = ctk.CTkImage(light_image=img_pil, size=(640, 480))

    # Cập nhật ảnh mới vào label (KHÔNG tạo label mới!)
    video_label.configure(image=img_ctk)
    video_label.image = img_ctk  # Giữ tham chiếu ảnh, tránh bị xóa

    window.after(10, update_frame)


# Xác nhận xóa toàn bộ
def clear_all_text():
    confirm = messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa toàn bộ nội dung không?")
    if confirm:
        recognized_text.set("")

# Xóa 1 ký tự cuối
def delete_last_character():
    current_text = recognized_text.get()
    if current_text:
        recognized_text.set(current_text[:-1])


# Hàm hiển thị thông báo ngắn
def show_simple_message(message, duration=2000):  # duration tính bằng ms
    notification_label.configure(text=message)
    notification_label.place(relx=0.5, rely=0.9, anchor="center")
    window.after(duration, lambda: notification_label.place_forget())

# Trong phần setup giao diện, bạn tạo 1 Label ẩn
notification_label = ctk.CTkLabel(window, text="", font=("Segoe UI", 14), text_color="red")
        


""" Hàm cho nút nhấn"""
#Nút bật tắt âm thanh
def toggle_camera():
    global cap, is_camera_on, tracking_active, result_textbox
    if is_camera_on:
        # Nút đang chuyển từ trạng thái bật sang tắt camera
        if cap:
            cap.release()
        cap = None
        is_camera_on = False
        #tracking_active = False
        #btn_toggle_recognize.configure(text="▶Start", text_color="#27ae60")
        # Cập nhật giao diện với CustomTkinter
        btn_toggle_camera.configure(text="📹Mở Camera", fg_color= button_color)
        #status_label.configure(text = "Đã tắt camera", text_color = "red")
        video_label.configure(image=empty_image_ctk)  # Dùng `CTkImage`
        if tracking_active:
            messagebox.showwarning("Thông báo", "Hãy mở camera để tiếp tục nhận diện")
        
    else:
        # Nút chuyển từ trạng thái tắt sang bật camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera")
            return
        
        is_camera_on = True
        btn_toggle_camera.configure(text="Tắt Camera", fg_color="#2c3e50")
        #status_label.configure(text = "Camera đã được mở", text_color = "#2c3e50")
        update_frame()

# Nút chọn chế độ
def option_detect():
    show_popup()


def show_popup():

    def on_select(option):
        global recognition_mode, result_textbox
        if option == "Nhận diện từng chữ cái":
            recognition_mode = "letter"
            status_label.configure(text= "Đang ở chế độ nhận diện từng chữ cái", text_color = "#2c3e50")
            result_textbox.configure(textvariable = single_char_text)
            recognized_text.set("")
            
            btn_clear_all.pack_forget()
            btn_space.pack_forget()
            btn_delete_one.pack_forget()
        if option == "Nhận diện chuỗi ký tự":
            recognition_mode = "sequence"
            status_label.configure(text= "Đang ở chế độ nhận diện chuỗi ký tự", text_color = "#2c3e50")
            result_textbox.configure(textvariable=recognized_text)
            btn_clear_all.pack(side="left", padx=10, fill = "x")
            btn_space.pack(side="left", padx=10, fill = "x")
            btn_delete_one.pack(side="left", padx=10, fill = "x")
        print(f"Chế độ đang chọn: {recognition_mode}")
        popup.destroy()
    

    # Tạo cửa sổ popup
    popup = ctk.CTkToplevel(window)
    popup.attributes("-topmost", True)
    popup.geometry("350x200")
    popup.title("Chọn chế độ")
    popup.configure(fg_color="#ffffff")

    # Tiêu đề
    label_title = ctk.CTkLabel(popup, text="Chọn chế độ:", font=("Segoe UI", 16, "bold"), text_color="#2c3e50")
    label_title.pack(pady=10)

    # Danh sách tùy chọn
    options = ["Nhận diện từng chữ cái", "Nhận diện chuỗi ký tự"]
    colors = ["#3498db", "#27ae60"]

    for i, option in enumerate(options):
        btn = ctk.CTkButton(popup, text=option, font=("Segoe UI", 14), corner_radius=25,
                            fg_color=colors[i], hover_color="#34495e",
                            text_color="white", width=180, height=40,
                            command=lambda opt=option: on_select(opt))
        btn.pack(pady=5)        

#Nút bật tắt âm thanh
def toggle_sound():
    global sound_enabled
    global is_camera_on, tracking_active
    if sound_enabled:
        # Nút đang ở trạng thái tắt âm thanh
        sound_enabled = False
        btn_toggle_sound.configure(text="🔊 Bật âm thanh")

    else:
        # Nút đang ở trạng thái bật âm
        if recognition_mode:
            if not is_camera_on:
                messagebox.showinfo("Thông báo", "Vui lòng mở camera")
            else:
                if not tracking_active:
                    messagebox.showinfo("Thông báo", "Vui lòng nhấn nút Start")
                else:
                    sound_enabled = True
                    btn_toggle_sound.configure(text="🔇 Tắt âm thanh", fg_color="#2c3e50")
        else:
            if not is_camera_on:
                messagebox.showinfo("Thông báo", "Vui lòng mở camera và chọn chế độ")
            else:
                if not tracking_active:
                    messagebox.showinfo("Thông báo", "Vui lòng chọn chế độ")

#Nút Start/Stop
def toggle_tracking():
    global tracking_active, notification_label
    global is_camera_on, recognition_mode
    if tracking_active:
        # Nút đang chuyển từ  trạng thái dừng nhận diện sang nhận diện
        confirm = messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn dừng nhận diện không?")
        if confirm:
            tracking_active = False

            # Cập nhật giao diện với CustomTkinter
            btn_toggle_recognize.configure(text="▶Start", fg_color = "#27ae60" )    

    else:
        # Nút đang chuyển từ trạng thái nhận diện 
        if not is_camera_on:
            if recognition_mode:
                messagebox.showinfo("Thông báo", "Vui lòng mở camera")
            else:  
                messagebox.showinfo("Thông báo", "Vui lòng mở camera và chọn chế độ")
    
        else:
            if recognition_mode:
                tracking_active = True

                btn_toggle_recognize.configure(text="⏹ Stop", fg_color="#e74c3c")
            else:
                messagebox.showwarning("Thông báo", "Vui lòng chọn chế độ")                    

#Nút Reset
def reset():
    global is_camera_on, sound_enabled, tracking_active, recognition_mode, recognized_text, result_textbox
    global cap

    if is_camera_on == True:
        if cap:
            cap.release()
        cap = None
        is_camera_on = False
    video_label.configure(image=empty_image_ctk)  # Hiển thị ảnh trắng       
    is_camera_on = False
    tracking_active = False
    recognition_mode = None
    sound_enabled = False
    btn_clear_all.pack_forget()
    btn_space.pack_forget()
    btn_delete_one.pack_forget()
    btn_toggle_camera.configure(text="📹Mở Camera")
    btn_toggle_recognize.configure(text="▶ Start")
    btn_toggle_sound.configure(text="🔊 Bật âm thanh")
    recognized_text.set("")
    single_char_text.set("")
    status_label.configure(text = "")

# Nút thoát
def exit_window():
    global is_camera_on, sound_enabled, tracking_active, recognition_mode, recognized_text, result_textbox
    global cap

    if is_camera_on:
        if cap is not None:
            cap.release()
        cap = None
        is_camera_on = False

    video_label.configure(image=empty_image_ctk)      
    tracking_active = False
    recognition_mode = None
    sound_enabled = False

    btn_clear_all.pack_forget()
    btn_space.pack_forget()
    btn_delete_one.pack_forget()

    recognized_text.set("")
    single_char_text.set("")
    status_label.configure(text="")

    window.destroy()

# Nút thêm khoảng trắng
def add_space():
    global space_added, last_detected_time, recognized_text, has_detected
    if has_detected:
        #if space_added == True: 
        recognized_text.set(recognized_text.get() + " ")
        last_detected_time = time.time() 
        space_added == False
        has_detected = False

threading.Thread(target=prediction_worker, daemon=True).start()
window.mainloop()
if cap:
    cap.release()
cv2.destroyAllWindows()

