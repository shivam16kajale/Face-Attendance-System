import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.set_page_config(layout="wide")
st.sidebar.title("Choose options")
options = st.sidebar.radio("", ['Register', 'Log In', 'Log Out', 'Attendance Sheet'], index=None)

@st.cache_resource(show_spinner=False)
def load_model(data_file):
    if not os.path.exists(data_file):
        return None
    df = pd.read_csv(data_file)
    if df.empty:
        return None
    df.dropna(inplace=True)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    final_pr_data = []
    for i in x.values:
        md = i.reshape(468, 3)
        center = md - md[1]
        distance = np.linalg.norm(md[33] - md[263])
        fpd = center / distance
        final_pr_data.append(fpd.flatten())
    x = pd.DataFrame(final_pr_data)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=79, max_depth=19, max_features="sqrt")
    return rf.fit(x_train, y_train)

class Face_Attendance_System:
    def __init__(self, data_file="complete_face_dataset.csv", attendance_file='Attendance_sheet.csv'):
        self.data_file = data_file
        self.attendance_file = attendance_file
        self.model = load_model(data_file)
        self.fm_model = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9
        )
        self.camera = st.empty()

    def register(self, name, samples=300):
        self.name = name
        vid = cv2.VideoCapture(0)
        face_data = []
        count = 0
        while count <= samples:
            s, frame = vid.read()
            if not s:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.camera.image(rgb, channels="RGB")
            result = self.fm_model.process(rgb)
            if result.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=rgb,
                    landmark_list=result.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
                face = [val for i in result.multi_face_landmarks[0].landmark for val in (i.x, i.y, i.z)]
                face.append(name)
                face_data.append(face)
                count += 1
                cv2.putText(frame, f"Sample taken {count}/{samples}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.camera.image(rgb, channels="RGB")
            if count == samples:
                st.caption("Samples collected")
                break
        self.camera.empty()
        vid.release()
        cv2.destroyAllWindows()
        df = pd.DataFrame(face_data)
        df.to_csv(self.data_file, mode='a', header=False, index=False)
        self.model = load_model(self.data_file)

    def detect_and_predict(self, rgb):
        result = self.fm_model.process(rgb)
        if result.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=rgb,
                landmark_list=result.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            face = [val for i in result.multi_face_landmarks[0].landmark for val in (i.x, i.y, i.z)]
            f1 = np.array(face).reshape(468, 3)
            center = f1 - f1[1]
            distance = np.linalg.norm(f1[33] - f1[263])
            fpd = center / distance
            get_face = [fpd.flatten()]
            return self.model.predict(get_face)[0] if self.model else None
        return None

    def login_to_system(self):
        if not self.model:
            st.warning("Model not trained. Register a user first.")
            return

        vid = cv2.VideoCapture(0)
        st.info("Camera is ON. Click the Login button to capture and predict.")
        self.camera_frame = st.empty()

        captured = False
        predicted_name = None

        while vid.isOpened() and not captured:
            ret, frame = vid.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.camera_frame.image(rgb, channels="RGB")

            if st.button("Login", key="login_button"):
                predicted_name = self.detect_and_predict(rgb)
                if predicted_name:
                    st.info(f"Predicted: {predicted_name}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes", key="confirm_yes"):
                            self.name = predicted_name
                            self.login_time = datetime.now().strftime('%I:%M:%S')
                            self.login_date = datetime.now().strftime("%d-%m-%Y")
                            st.success(f"{self.name} login successful")
                            self.excelbook()
                            captured = True
                    with col2:
                        if st.button("No", key="confirm_no"):
                            st.warning("Wrong prediction. Try again by clicking Login.")
                else:
                    st.warning("Face not detected. Please try again.")
        vid.release()
        cv2.destroyAllWindows()

    def logout_from_system(self):
        if not self.model:
            st.warning("Model not trained. Register a user first.")
            return
        vid = cv2.VideoCapture(0)
        logout_success = False
        if st.button("Log Out"):
            while not logout_success:
                s, frame = vid.read()
                if not s:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.camera.image(rgb, channels="RGB")
                name = self.detect_and_predict(rgb)
                if name:
                    self.name = name
                    self.logout_time = datetime.now().strftime('%I:%M:%S')
                    self.logout_date = datetime.now().strftime("%d-%m-%Y")
                    st.success(f"{self.name} logged out successfully")
                    self.excelbook()
                    logout_success = True
        vid.release()
        cv2.destroyAllWindows()

        
    def excelbook(self):
        file = self.attendance_file
        today = datetime.now().strftime("%d-%m-%Y")
        columns = ["Date", "Name", "Login_time", "Logout_time", "Work_Hours"]
        df = pd.read_csv(file) if os.path.exists(file) else pd.DataFrame(columns=columns)

        if hasattr(self, "login_time") and not hasattr(self, "logout_time"):
            value = (df["Name"] == self.name) & (df["Date"] == today)
            if not value.any():
                new_row = {
                    "Date": today,
                    "Name": self.name,
                    "Login_time": self.login_time,
                    "Logout_time": "",
                    "Work_Hours": ""
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        if hasattr(self, "logout_time"):
            value = (df["Name"] == self.name) & (df["Date"] == today)
            if value.any():
                df.loc[value, "Logout_time"] = self.logout_time
                try:
                    login_time = datetime.strptime(df.loc[value, "Login_time"].values[0], "%I:%M:%S")
                    logout_time = datetime.strptime(self.logout_time, "%I:%M:%S")
                    df.loc[value, "Work_Hours"] = str(logout_time - login_time)
                except Exception as e:
                    print("Error in work hour calculation:", e)
        df.to_csv(file, index=False)


start = Face_Attendance_System()
if options == "Register":
    st.title("Face Registration")
    name = st.text_input("Enter Name for registration")
    if name:
        start.register(name)
        st.success(f"{name} registered successfully")

elif options == "Log In":
    start.login_to_system()

elif options == "Log Out":
    start.logout_from_system()

elif options == "Attendance Sheet":
    if os.path.exists("Attendance_sheet.csv"):
        df = pd.read_csv("Attendance_sheet.csv")
        st.dataframe(df)
        data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download File", file_name="Attendance_sheet.csv", mime="text/csv", data=data)
    else:
        st.write("No data available.")
else:
    st.markdown("""
    # 👋 Welcome to the Face Recognition Attendance System

    Built with Object-Oriented Programming (OOP) and real-time face detection.

    - Register new users
    - Log in via face recognition
    - Log out and calculate working hours
    - Save attendance to a CSV
    """)
