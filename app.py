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

options=st.sidebar.radio("",['Register','Log In','Log Out','Attendance Sheet'],index=None)





class Face_Attendance_System:
    def __init__(self,data_file = "complete_face_dataset.csv",attendance_file = 'Attendance_sheet.csv'):
        print("code started.....")
        print("Welcome to the lab:")
        self.data_file = data_file
        self.attendance_file = attendance_file
        self.model=None
        self.fm_model=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
        max_num_faces=1, 
        refine_landmarks=False,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9)
        self.camera = st.empty()
        self.train_model()
        

    def register(self,name,samples=300):
        self.name=name
        self.samples = samples
        print("face registration started")
        vid = cv2.VideoCapture(0)
        

        # num_registers = int(input("Enter number of registrations: "))
        face_data=[]
        # for i in range(num_registers):
        # name = input("Enter name for registration:")
        # name=st.text_input("Enter Name: ")
        self.count = 0
        while self.count<=self.samples:
            s,frame = vid.read()
            if s==False:
                break
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            self.camera.image(rgb,channels="RGB")
            result = self.fm_model.process(rgb)
            if result.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image = rgb,landmark_list=result.multi_face_landmarks[0],connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                                            landmark_drawing_spec=None,connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

                face=[]
                if result.multi_face_landmarks:
                    for i in result.multi_face_landmarks[0].landmark:
                        face.append(i.x)
                        face.append(i.y)
                        face.append(i.z)
                cv2.putText(frame,f"Sample taken {self.count}/{self.samples}",(30,80),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                face.append(self.name)
                face_data.append(face)
                self.count+=1
            # cv2.imshow("img",frame)
            self.camera.image(rgb,channels="RGB")
            if self.count==300:
                st.caption("Samples collected")
                break
        # print(f"{name} is register successfully.")
        # st.write(f"{self.name} is register successfully.")

            
        # print(f"{num_registers} are registered success...")
        self.camera.empty()
        vid.release()
        cv2.destroyAllWindows()
        df= pd.DataFrame(face_data)
        df.to_csv("complete_face_dataset.csv",mode='a',header=None,index=False)
        # print(f"{num_registers} are added in database..")
        self.train_model()
        
    def train_model(self):
        df= pd.read_csv(self.data_file)
        df.dropna(inplace=True)
        x=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        #preprocessing for location in variant
        final_pr_data =[]
        for i in x.values:
            md=i.reshape(468,3)
            center = md - i.reshape(468,3)[1]
            distance = np.linalg.norm(i.reshape(468,3)[33]-i.reshape(468,3)[263])
            fpd = center/distance
            final_pr_data.append(fpd.flatten())
        x=pd.DataFrame(final_pr_data)
        
        #spliting data
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

        # model 
        rf = RandomForestClassifier(n_estimators = 79,max_depth=19,max_features="sqrt")
        self.model=rf.fit(x_train,y_train)
        print("model train successfully....")

    

    
    def login_to_system(self):
        vid=cv2.VideoCapture(0)
        Login = st.button("Login")
        login_success = False
        while True:
                s,f=vid.read()
                if s==False:
                    break
                rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
            
                result=self.fm_model.process(rgb)
            
                if result.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks (image = f,landmark_list=result.multi_face_landmarks[0],connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                    
                self.camera.image(rgb,channels="RGB")
            
                key = cv2.waitKey(1) & 255
                if key == ord("q"):
                    break
                face=[]
                if Login and not login_success:
                    if result.multi_face_landmarks:
                        for i in result.multi_face_landmarks[0].landmark:
                            face.append(i.x)
                            face.append(i.y)
                            face.append(i.z)

                
                if bool(face)!=False:
                    get_face = []
                    f1=np.array(face).reshape(468,3)
                    center = f1 - f1[1]
                    distance = np.linalg.norm(f1[33]-f1[263])
                    fpd = center/distance
                    get_face.append(fpd.flatten())

                    # prediction
                    self.name = self.model.predict(get_face)[0]
                    st.success(f"{self.name} login successfull")
                    self.login_time = datetime.now().strftime('%I:%M:%S')
                    self.login_date = datetime.now().strftime("%d-%m-%Y")
                    login_success = True
                    self.excelbook()
                    # print(f"Login successfull for {self.name}")
                    # print(f"Login Time:{self.login_time}")
                    # print(f"login date: {self.login_date}")
        
                
        vid.release()
        cv2.destroyAllWindows()
        
        



    def logout_from_system(self):
        vid=cv2.VideoCapture(0)
        logout = st.button("Log Out")
        logout_success = False
       

        while True:
                s,f=vid.read()
                if s==False:
                    break
                rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
            
                result=self.fm_model.process(rgb)
            
                if result.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks (image = f,landmark_list=result.multi_face_landmarks[0],connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                    
                # cv2.imshow("image",f)
                self.camera.image(f,channels="RGB")
            
                key = cv2.waitKey(1) & 255
                if key == ord("q"):
                    break
                face=[]
                if logout and not logout_success:
                    if result.multi_face_landmarks:
                        for i in result.multi_face_landmarks[0].landmark:
                            face.append(i.x)
                            face.append(i.y)
                            face.append(i.z)

                
                if bool(face)!=False:
                    get_face = []
                    f1=np.array(face).reshape(468,3)
                    center = f1 - f1[1]
                    distance = np.linalg.norm(f1[33]-f1[263])
                    fpd = center/distance
                    get_face.append(fpd.flatten())

                    # prediction
                    self.name = self.model.predict(get_face)[0]
                    st.success(f"{self.name} is logged out successfully")
                    self.logout_time = datetime.now().strftime('%I:%M:%S')
                    self.logout_date = datetime.now().strftime("%d-%m-%Y")
                    print(f"Log out successfull for {self.name}")
                    print(f"Logout Time:{self.logout_time}")
                    print(f"Logout date: {self.logout_date}")
                    logout_success=True
                    self.excelbook()

        
        vid.release()
        cv2.destroyAllWindows()
        
       

    

    def excelbook(self):

        file = self.attendance_file
        columns = ["Date", "Name", "Login_time", "Logout_time", "Work_Hours"]
        today = datetime.now().strftime("%d-%m-%Y")

        # Create file or load existing one
        if os.path.exists(file):
            df = pd.read_csv(file)
        else:
            df = pd.DataFrame(columns=columns)

        
        if hasattr(self, "login_time") and not hasattr(self, "logout_time"):
            value = (df["Name"] == self.name) & (df["Date"] == today)
            if not value.any():
                # Create complete login row
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
                    login_time_str = df.loc[value, "Login_time"].values[0]
                    login_time = datetime.strptime(login_time_str, "%I:%M:%S")
                    logout_time = datetime.strptime(self.logout_time, "%I:%M:%S")
                    work_hours = str(logout_time - login_time)
                    df.loc[value, "Work_Hours"] = work_hours
                except Exception as e:
                    print("Error in work hour calculation:", e)

        # Save all updates to the file
        df.to_csv(file, index=False)





start = Face_Attendance_System()
# print("""1. Register Face
# 2. Login
# 3. Logout 
# 4. Exit""")
# op = int(input("What do you want to do: "))
# while True:
#     if op == 1:
#         start.register()
#     elif op == 2:
#         start.login_to_system()
#     elif op == 3:
#         start.logout_from_system()
#     elif op == 4:
#         exit()
#     op = int(input("What do you want to do: "))
def home():
    st.markdown("""
# 👋 Welcome to the Face Recognition Attendance System

This project is built to solve a simple but common problem:  
**Taking attendance — quickly, accurately, and without touching anything.**

Instead of signing a sheet or scanning a card, this system uses your **face** to:

- 🧍‍♂️ Register you into the system  
- 🔐 Log your attendance when you arrive  
- 🚪 Log you out when you leave  
- 🗂️ Save everything automatically into a CSV file

---

### 💡 What's Different About This Project?

Most AI demos stop at the "cool detection part."  
This one is built like **real software** — using **Object-Oriented Programming (OOP)** to keep the code clean, modular, and scalable.

Everything is handled through a single class:  
```python
class Face_Attendance_System"""
)
if options == "Register":
    st.title("Face Registration ")
    name = st.text_input("Enter Name for registration")
    if name:
       start.register(name)
       
       st.write(f"{name} register successfully")
    
       st.success("Model Train on Updated data..")

elif options =="Attendance Sheet":
    if os.path.exists("Attendance_sheet.csv"):
    
        df = pd.read_csv("Attendance_sheet.csv")
        st.dataframe(df)
        data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download File",file_name="Attendance_sheet.csv",mime="text/csv",data=data)
    else:
        st.write("No data available.....")

elif options=="Log In":
    start.login_to_system()

elif options=="Log Out":
    start.logout_from_system()
else:
    st.markdown("""
# 👋 Welcome to the Face Recognition Attendance System

This project is built to solve a simple but common problem:  
**Taking attendance — quickly, accurately, and without touching anything.**

Instead of signing a sheet or scanning a card, this system uses your **face** to:

- 🧍‍♂️ Register you into the system  
- 🔐 Log your attendance when you arrive  
- 🚪 Log you out when you leave  
- 🗂️ Save everything automatically into a CSV file

---

### 💡 What's Different About This Project?

Most AI demos stop at the "cool detection part."  
This one is built like **real software** — using **Object-Oriented Programming (OOP)** to keep the code clean, modular, and scalable.

Everything is handled through a single class:  
```python
class Face_Attendance_System"""
)
    
