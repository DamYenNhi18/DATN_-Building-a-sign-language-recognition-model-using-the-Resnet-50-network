
import cv2
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf
import os
import time
last_save_time = 0
#save_interval = 0.5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

imgSize = 224
save_folder = r"D:\DoAnTotNghiep\data_5\nothing2" # Thư mục lưu ảnh
os.makedirs(save_folder, exist_ok=True)  # Tạo thư mục nếu chưa có

saving_enabled = False  # Biến kiểm soát việc lưu ảnh
max_images = 3000 # Số ảnh tối đa cần lưu
image_count = len(os.listdir(save_folder))  # Kiểm tra số ảnh đã có


while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = img.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            #mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Vẽ khung bao quanh bàn tay
            cv2.rectangle(img, (x_min -20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)

            imCrop = img[y_min-20:y_max+20, x_min-20:x_max+20]
            if imCrop.size == 0:
                continue  # Nếu ảnh trống, bỏ qua để tránh lỗi

            # Tạo nền trắng
            imWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Xử lý tỷ lệ ảnh
            h_crop, w_crop = imCrop.shape[:2]
            aspectRatio = h_crop / w_crop

            try:
                if aspectRatio > 1:  # Chiều cao lớn hơn chiều rộng
                    k = imgSize / h_crop
                    wCal = math.ceil(k * w_crop)
                    imgResize = cv2.resize(imCrop, (wCal, imgSize))
                    w_gap = (imgSize - wCal) // 2

                    if w_gap >= 0 and w_gap + wCal <= imgSize:
                        imWhite[:, w_gap:w_gap + wCal] = imgResize
                    else:
                        print("⚠️ Kích thước ảnh vượt quá imgSize. Bỏ qua.")
                        continue


                else:  # Chiều rộng lớn hơn chiều cao
                    k = imgSize / w_crop
                    hCal = math.ceil(k * h_crop)
                    imgResize = cv2.resize(imCrop, (imgSize, hCal))
                    h_gap = (imgSize - hCal) // 2

                    if h_gap >= 0 and h_gap + hCal <= imgSize:
                        imWhite[h_gap:h_gap + hCal, :] = imgResize
                    else:
                        print("⚠️ Kích thước ảnh vượt quá imgSize. Bỏ qua.")
                        continue
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý ảnh: {e}")
                continue

            # Hiển thị kết quả
            cv2.imshow("Image Crop", imCrop)
            cv2.imshow("Image White", imWhite)

            # Lưu ảnh khi nhấn 'S' và chưa đủ 1000 ảnh
            current_time = time.time()
            if saving_enabled and image_count < max_images:
                if current_time - last_save_time >= 1:
                    image_count += 1
                    image_path = os.path.join(save_folder, f"{image_count}.jpg")
                    cv2.imwrite(image_path, imWhite)
                    print(f"📸 Đã lưu: {image_count}/{max_images}")

            # Nếu đạt 1000 ảnh thì tự động dừng lưu
            if image_count >= max_images:
                saving_enabled = False
                print("✅ Đã lưu đủ ảnh, tự động dừng.")

    cv2.imshow("Hand Tracking", img)

    # Kiểm tra phím bấm
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and image_count < max_images:  
        saving_enabled = not saving_enabled  # Nhấn S để bật/tắt lưu ảnh
        if saving_enabled:
            print("🔴 Bắt đầu lưu ảnh...")
        else:
            print("⏹ Dừng lưu ảnh.")
    elif key == ord('q'):  
        break  # Thoát chương trình

cap.release()
cv2.destroyAllWindows()
