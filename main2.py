################################################################################
##
## BY: WANDERSON M.PIMENTA
## PROJECT MADE WITH: Qt Designer and PySide2
## V: 1.0.0
##
## This project can be used freely for all uses, as long as they maintain the
## respective credits only in the Python scripts, any information in the visual
## interface (GUI) can be modified without any implication.
##
## There are limitations on Qt licenses if you want to use your products
## commercially, I recommend reading them on the official website:
## https://doc.qt.io/qtforpython/licenses.html
##
################################################################################
#python main.py --shape-predictor shape_predictor_68_face_landmarks.dat
import sys
import platform
import winsound
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
import cv2
import psutil
# GUI FILE
from imutils.video import VideoStream
from imutils import resize
from imutils import face_utils
from scipy.spatial import distance as dist
import argparse
from random import randint
import math
import time
import dlib
#import tensorflow as tf
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from app_modules import *
from threading import Thread
import qimage2ndarray
ap = argparse.ArgumentParser()
from PyQt5 import QtGui as qt5g
from PyQt5.QtCore import QThread,pyqtSignal
# initialize dlib's face detector (HOG-based) 
# and then create the facial landmark predictor
from queue import Queue 
from threading import Thread 
from playsound import playsound
'''def modelf():
    yaml_file = open('/content/drive/My Drive/test/final_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = tf.keras.models.model_from_yaml(loaded_model_yaml)
# load weights into new model
    loaded_model.load_weights("/content/drive/My Drive/test/final_model.h5")
    #print("Loaded model from disk")
    loaded_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
def use(image,loaded_model):
    img = cv2.resize(img,(100,100))
    img = np.reshape(img,[1,100,100,3])
    classes = loaded_model.predict_classes(img)
    if classes==1:
        return True
    else:
        return False'''
def awake():
    playsound('Air Horn-SoundBible.com-964603082.mp3')
def alert():
    playsound('School_Fire_Alarm-Cullen_Card-202875844.mp3')    
class Threadcam(object):
    
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,2)
        self.cap.set(3,640)
        self.cap.set(4,480)
        self.cap.set(5,30)
        self.FPS=1/30
        (self.status,self.frame)=self.cap.read()
        self.thread=Thread(target=self.update, args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while True:
            (self.status,self.frame)=self.cap.read()
            #time.sleep(self.FPS)
    def  retframe(self):
        return self.frame
def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[9])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[7])  # 53, 57
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    # i.append(indice)
    # return the mouth aspect ratio
    return mar
def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])  # 51, 59
    B = dist.euclidean(eye[2], eye[4])  # 53, 57
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])  # 49, 55
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # i.append(indice)
    # return the eye aspect ratio
    return ear
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)
(eStart, eEnd) = (37, 46)
MOUTH_AR_THRESH = 0.76
stilopen=False
buffimage=0
class YouThread(QThread):
    result=pyqtSignal(object)
    def __init__(self,image,detector,predictor):
        QThread.__init__(self)
        self.image = image
        self.predictor = predictor
        self.detector = detector
        

    def run(self):
        faces = self.detector(self.image, 0)
        self.i=0
        maxwidth=0  
        arrayface=[]
        (b1,b2,b3,b4,b5)=(0,0,0,0,0) 
        mar=0
        ear=0
        for face in faces:
            shape = self.predictor(self.image, face)
            shape = face_utils.shape_to_np(shape)
            (bX1, bY1, bW1, bH1) = face_utils.rect_to_bb(face)
            if bW1>maxwidth:
                maxwidth=bW1
                b1=bX1
                b2=bY1
                b3=bW1
                b4=bH1
                b5=face
                mouth = shape[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)
                eye = shape[rStart:rEnd]
                ear = eye_aspect_ratio(eye)
        self.result.emit([b1,b2,b3,b4,b5,ear,mar])
 
    
def find_face(q,image,detector,predictor):
    self.arrayface=[]
    faces = detector(image, 0)
    i=0
    maxwidth=0
    for face in faces:
        shape = predictor(image, face)
        shape = face_utils.shape_to_np(shape)
        (bX, bY, bW, bH) = face_utils.rect_to_bb(face)
        if abs(bW)>maxwidth:
            maxwidth=abs(bW)
            self.self.arrayface.clear()
            self.arrayface.append(bX)
            self.arrayface.append(bY)
            self.arrayface.append(bW)
            self.arrayface.append(bH)
            self.arrayface.append(face)
    q.put(self.arrayface)
        
def calc(image,face,q,q2,detector,predictor):
    global contor
    # Read the next frame from the stream in a different thread
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # get image infos

    ear=0
    mar=0
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #faces = detector(image, 0)
    #for face in faces:
    shape = predictor(image, face)
    shape = face_utils.shape_to_np(shape)
    mouth = shape[mStart:mEnd]
    mar = mouth_aspect_ratio(mouth)

    eyes = shape[rStart:rEnd]
    ear = eye_aspect_ratio(eyes)
    (bX, bY, bW, bH) = face_utils.rect_to_bb(face)
    #cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)

    #break
    q.put(mar)
    q2.put(ear)
    #q3.put(image)

class MainWindow(QMainWindow):

    econtor=0
    '''def find_face():
        self.self.arrayface=[]
        faces = self.detector(image, 0)
        i=0
        maxwidth=0
        for face in faces:
            shape = self.predictor(self.image, face)
            shape = face_utils.shape_to_np(shape)
            (bX, bY, bW, bH) = face_utils.rect_to_bb(face)
            if bW>maxwidth:
                maxwidth=bW
                self.self.arrayface.clear()
                self.self.arrayface.append(bX)
                self.self.arrayface.append(bY)
                self.self.arrayface.append(bW)
                self.self.arrayface.append(bH)
                self.self.arrayface.append(face)
        cv2.rectangle(self.image, (self.arrayface[0], self.arrayface[1]), (self.arrayface[0] + self.arrayface[2], self.arrayface[1] + self.arrayface[3]),(0, 255, 0), 1)'''
    def get_result(self,arra):
        if arra[4]:
            cv2.rectangle(self.image, (arra[0], arra[1]), (arra[0] + arra[2], arra[1] + arra[3]),(0, 255, 0), 1)
            self.curEAR=arra[5]
            self.curMAR=arra[6]
            if arra[5]<0.20:
                self.ui.eye_2.setText("Eye State : <font color='#d50000'>Close</font>")       
                if self.contor>=3:
                    self.re=True
                else:
                    self.contor=self.contor+1    
            else: 
                if self.contor==-30 and self.re:
                    self.re=False
                    self.contor=0
                else:
                    if self.contor>0 and not self.re:
                        self.contor=self.contor-1   
                    if self.contor>-30 and self.re:
                        self.contor=self.contor-1
                self.ui.eye_2.setText("Eye State : <font color='#14FFEC'>Open</font>")  
            
            if arra[6]>=MOUTH_AR_THRESH:
                self.ui.mouth.setText("Mouth : <font color='#d50000'>Widely open</font>")
                if self.mcontor>=5 :
                    self.rm=True
                    self.T2 = Thread(target=alert) # create thread
                    #p = multiprocessing.Process(target=playsound, args=("file.mp3",))
                    #p.start()
                    #input("press ENTER to stop playback")
                    #time.sleep(1.0)
                    #p.terminate()
                    self.T2.start() # Launch created thread
                    #winsound.PlaySound(r'C:\Users\DELL\Desktop\projet_app\School_Fire_Alarm-Cullen_Card-202875844.mp3', winsound.SND_ASYNC)
                else:
                    self.mcontor=self.mcontor+1
            else:
                self.ui.mouth.setText("Mouth : <font color='#14FFEC'>Normal</font>")
                if self.mcontor==-30 and self.rm:
                    self.rm=False
                    #winsound.PlaySound(None, winsound.SND_PURGE)
                    self.mcontor=0
                else:
                    if self.mcontor>0 and not self.rm:
                        self.mcontor=self.mcontor-1
                    if self.mcontor>-30 and self.rm:
                        self.mcontor=self.mcontor-1    
        if self.re:
            self.ui.general.setText("General status : <font color='#d50000'>Sleepy</font>")
            
            T = Thread(target=awake) # create thread
            T.start() # Launch created thread
        else:
            if self.rm:
                self.ui.general.setText("General status: <font color='#fdd835'> Yawning</font>")
               
            else:
                self.ui.general.setText("General Status : <font color='#14FFEC'>Normal</font>")    
    def showface(self):
        '''p = Thread(target = calc, args =(image,arrayface[4],self.q2,self.q3,self.detector,self.predictor ))
            p.start()
            p.join()
            self.curMAR=self.q2.get()
        
            self.curEAR=self.q3.get()
            if self.curEAR<0.2:
                self.ui.eye_2.setText("Eye State : <font color='#d50000'>Close</font>")
            else:
                self.ui.eye_2.setText("Eye State : <font color='#14FFEC'>Open</font>")   
            if self.curMAR>=MOUTH_AR_THRESH:
                self.contor=self.contor+1
                self.ui.mouth.setText("Mouth : <font color='#d50000'>Widely open</font>")
            else:
                self.contor=0
                self.ui.mouth.setText("Mouth : <font color='#14FFEC'>Normal</font>")
                if self.mcontor>0:
                    self.ui.general.setText("General status : <font color='fdd835'>Yawning</font>")
                    self.mcontor=self.mcontor-1
                else:
                    self.ui.general.setText("General Status : <font color='#14FFEC'>Normal</font>")    
            if self.contor>=30:
                self.mcontor=90
                print("here")
                self.ui.general.setText("General status: <font color='fdd835'> Yawning</font>")'''
        
        
        height, width, channel = self.image.shape
        step = channel * width
       
        qImg = QImage(self.image.data, width, height, step, QImage.Format_RGB888)
        self.ui.label_6.setPixmap(QPixmap.fromImage(qImg))
        self.ui.label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.label_20.setPixmap(QPixmap.fromImage(qImg))

        
    def disp(self):
        #ret,frame=self.cap.read()
        #time.sleep(1/30)
        frame=self.thre.retframe()
        
        #image=qimage2ndarray.array2qimage(frame)
        rows, cols, _ = frame.shape
        gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

        _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        
        
        self.image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.waitKey(self.FPS_MS)
        '''p1=Thread(target=find_face, args=(self.q,image,self.detector,self.predictor))
        p1.start()
        p1.join()
        self.arrayface=self.q.get()'''
        '''self.timer2 = QTimer()
        self.timer2.setInterval(1000)
        self.timer2.timeout.connect(self.recurring_timer)
        self.timer2.start()'''
        self.thread = YouThread(self.image,self.detector,self.predictor)
        self.thread.start()
        self.thread.result.connect(self.get_result)
        #arrayface=self.thread.res()
        self.thread.finished.connect(self.showface)            
    def update_plot_data(self):
    
        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        self.y = self.y[1:]  # Remove the first 
        self.y.append( self.curEAR)  # Add a new random value.

        self.data_line.setData(self.x, self.y)  # Update the data.
    def update_plot_data2(self):
        
        self.x2 = self.x2[1:]  # Remove the first y element.
        self.x2.append(self.x2[-1] + 1)  # Add a new value 1 higher than the last.

        self.y2 = self.y2[1:]  # Remove the first 
        self.y2.append( self.curMAR)  # Add a new random value.

        self.data_line2.setData(self.x2, self.y2)  # Update the data.    
    def __init__(self):
        self.check1=True
        self.check2=True
        self.check3=True
        self.curEAR=0
        self.curMAR=0
        self.arrayface=[]
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        time.sleep(2.0)
        self.q = Queue()
        self.q2 = Queue()
        self.q3 = Queue()
        self.FPS_MS=int((1/30)*1000)
        self.contor=0
        self.mcontor=0
        self.re=False
        self.rm=False
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow2()
        self.ui.setupUi(self)
        self.ui.btn_open_file.clicked.connect(self.Button)
        self.ui.btn_open_file.setStyleSheet("border-right: 5px solid rgb(20, 255, 236);")
        
        
        self.ui.btn_mouth.clicked.connect(self.Button)
        self.ui.pushButton_2.clicked.connect(self.Button)
        self.ui.btn_new_user.clicked.connect(self.Button)
        self.ui.btn_settings.clicked.connect(self.Button)
        
        
        
        self.graphWidget = pg.PlotWidget()
        self.ui.verticalLayout_14.addWidget(self.graphWidget,0,0,1,1)

        self.x = list(range(100))  # 100 time points
        self.y = [0 for _ in range(100)]  # 100 data points

        self.graphWidget.setBackground((20,20,20,0))
        self.graphWidget.setTitle("Eye aspect ratio curve", color="b", size="30pt")
        styles = {"color": "#f00", "font-size": "20px"}
        self.graphWidget.setLabel("left", "EAR", **styles)
        self.graphWidget.setLabel("bottom", "Time(fps)", **styles)
        pen = pg.mkPen(color=(25, 255, 236))
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
        
        
        
        
        self.graphWidget2 = pg.PlotWidget()
        self.ui.verticalLayout_10.addWidget(self.graphWidget2,0,0,1,1)

        self.x2 = list(range(100))  # 100 time points
        self.y2 = [0 for _ in range(100)]  # 100 data points

        self.graphWidget2.setBackground((20,20,20,0))

        pen2 = pg.mkPen(color=(25, 255, 236))
        self.data_line2 =  self.graphWidget2.plot(self.x2, self.y2, pen=pen2)
        
        
        self.thre=Threadcam()
        #self.cap.set(5,30)
        self.timer = QTimer()
        self.timer.timeout.connect(self.disp)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.timeout.connect(self.update_plot_data2)
        self.timer.start(60)
        #self.ui.btn_new_user.clicked.connect(eve)
        #self.ui.btn_new_user.installEventFilter(self)
        # set timer timeout callback function
        #self.timer.timeout.connect(self.viewCam)
        #QTimer.singleShot(2,self.controlTimer)
        #th=Thread(self)
        #th.changePixmap.connect(self.setImage)
        #th.start()
        ## PRINT ==> SYSTEM
        print('System: ' + platform.system())
        print('Version: ' +platform.release())

        ########################################################################
        ## START - WINDOW ATTRIBUTES
        ########################################################################

        ## REMOVE ==> STANDARD TITLE BAR
        UIFunctions.removeTitleBar(True)
        ## ==> END ##

        ## SET ==> WINDOW TITLE
        self.setWindowTitle('Main Window - Python Base')
        UIFunctions.labelTitle(self, 'Main Window - Python Base')
        UIFunctions.labelDescription(self, 'Current Window')
        ## ==> END ##

        ## REMOVE ==> STANDARD TITLE BAR
        #startSize = QSize(1000, 720)
        #self.resize(startSize)
        #self.setMinimumSize(startSize)
        #UIFunctions.enableMaximumSize(self, 500, 720)
        ## ==> END ##

        ## ==> CREATE MENUS
        ########################################################################

        ## ==> TOGGLE MENU SIZE
        self.ui.btn_toggle_menu.clicked.connect(lambda: UIFunctions.toggleMenu(self, 220, True))
        ## ==> END ##

        ## ==> ADD CUSTOM MENUS
        self.ui.stackedWidget.setMinimumWidth(20)
        UIFunctions.addNewMenu(self, "HOME", "btn_open_file", "url(:/16x16/icons/16x16/cil-home.png)", True)
        #UIFunctions.addNewMenu(self, "EYE", "btn_new_user", "url(:/16x16/icons/16x16/cil-eye-open.png)", True)
        #UIFunctions.addNewMenu(self, "Mouth", "btn_mouth", "url(:/16x16/icons/16x16/cil-mouth.png)", True)
        #UIFunctions.addNewMenu(self, "SETTINGS", "btn_settings", "url(:/20x20/icons/20x20/cil-settings.png)", False)
        ## ==> END ##

        # START MENU => SELECTION
        UIFunctions.selectStandardMenu(self, "btn_open_file")
        ## ==> END ##

        ## ==> START PAGE
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
        ## ==> END ##

        ## USER ICON ==> SHOW HIDE
        UIFunctions.userIcon(self, "FD", "", True)
        ## ==> END ##

        def eventFilter(self, source, event):
            if (event.type() == QtCore.QEvent.MouseButtonPress and source is self.btn_new_user):
                print('mouse-move:')
            return QMainWindow.eventFilter(self, source, event)
        ## ==> MOVE WINDOW / MAXIMIZE / RESTORE
        ########################################################################
        def moveWindow(event):
            # IF MAXIMIZED CHANGE TO NORMAL
            if UIFunctions.returStatus() == 1:
                UIFunctions.maximize_restore(self)

            # MOVE WINDOW
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # WIDGET TO MOVE
        self.ui.frame_label_top_btns.mouseMoveEvent = moveWindow
        ## ==> END ##

        ## ==> LOAD DEFINITIONS
        ########################################################################
        UIFunctions.uiDefinitions(self)
        ## ==> END ##

        ########################################################################
        ## END - WINDOW ATTRIBUTES
        ############################## ---/--/--- ##############################




        ########################################################################
        #                                                                      #
        ## START -------------- WIDGETS FUNCTIONS/PARAMETERS ---------------- ##
        #                                                                      #
        ## ==> USER CODES BELLOW                                              ##
        ########################################################################



        ## ==> QTableWidget RARAMETERS
        ########################################################################
        # ==> END ##



        ########################################################################
        #                                                                      #
        ## END --------------- WIDGETS FUNCTIONS/PARAMETERS ----------------- ##
        #                                                                      #
        ############################## ---/--/--- ##############################


        ## SHOW ==> MAIN WINDOW
        ########################################################################
        self.showMaximized()
        print(psutil.cpu_percent())
        ## ==> END ##

    ########################################################################
    ## MENUS ==> DYNAMIC MENUS FUNCTIONS
    ########################################################################
    def Button(self):
        # GET BT CLICKED
        btnWidget = self.sender()

        # PAGE HOME
        if btnWidget.objectName() == "btn_open_file":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            self.ui.page_home.show()
            UIFunctions.resetStyle(self, "btn_open_file")
            UIFunctions.labelPage(self, "Home")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))
        if btnWidget.objectName() == "btn_mouth":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_mouth)
            self.ui.page_mouth.show()
            UIFunctions.resetStyle(self, "btn_mouth")
            UIFunctions.labelPage(self, "Mouth State")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE NEW USER
        if btnWidget.objectName() == "btn_new_user":
            self.ui.stackedWidget.setCurrentWidget(self.ui.eye)
            UIFunctions.resetStyle(self, "btn_new_user")
            UIFunctions.labelPage(self, "Eye State")
            self.timer.stop()
            #self.ui.page_home.hide()
            #self.ui.eye.show()
            self.timer.start(20)
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE WIDGETS
        if btnWidget.objectName() == "btn_settings":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_settings)
            
            UIFunctions.resetStyle(self, "btn_settings")
            UIFunctions.labelPage(self, "Settings")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))
        
        
        if btnWidget.objectName() == "pushButton_2":
            #if self.ui.btn_mouth.isEnabled():
            '''bt=QObject.findChild(QPushButton,"btn_mouth")
            bt.setEnabled(False)
            print('hy')
            QTimer.singleShot(5000, lambda: bt.setDisabled(True))'''
            #for checkstate in self.findChildren(QPushButton):
            #    print(f'get check state:{checkstate.checkState()}')
            if self.check1:
                self.ui.mouth.setVisible(False)
                self.ui.btn_mouth.setVisible(False)
                self.check1=False
            else:
                self.ui.mouth.setVisible(True)
                self.ui.btn_mouth.setVisible(True)
                self.check1=True
                #time.sleep(2.0)
            #else:
            #    self.ui.btn_mouth.setEnabled(True)
    ## ==> END ##

    ########################################################################
    ## START ==> APP EVENTS
    ########################################################################

    ## EVENT ==> MOUSE DOUBLE CLICK
    ########################################################################
    #def eventFilter(self, watched, event):
    #    if watched == self.le and event.type() == QtCore.QEvent.MouseButtonDblClick:
    #        print("pos: ", event.pos())
    ## ==> END ##

    ## EVENT ==> MOUSE CLICK
    ########################################################################
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')
        if event.buttons() == Qt.MidButton:
            print('Mouse click: MIDDLE BUTTON') 
    ## ==> END ##

    ## EVENT ==> KEY PRESSED
    ########################################################################
    def keyPressEvent(self, event):
        print('Key: ' + str(event.key()) + ' | Text Press: ' + str(event.text()))
    ## ==> END ##

    ## EVENT ==> RESIZE EVENT
    ########################################################################
    def resizeEvent(self, event):
        self.resizeFunction()
        return super(MainWindow, self).resizeEvent(event)

    def resizeFunction(self):
        print('Height: ' + str(self.height()) + ' | Width: ' + str(self.width()))
    ## ==> END ##

    ########################################################################
    ## END ==> APP EVENTS
    ############################## ---/--/--- ##############################

if __name__ == "__main__":
    

    
    app = QApplication(sys.argv)
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeui.ttf')
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeuib.ttf')
    window = MainWindow()
    
    sys.exit(app.exec_())


