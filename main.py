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
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent,QTimer)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient,QImage)
from PySide2.QtWidgets import *
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
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from app_modules import *

(mStart, mEnd) = (49, 68)
(eStart, eEnd) = (37, 46)
MOUTH_AR_THRESH = 0.8

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

# initialize dlib's face detector (HOG-based) 
# and then create the facial landmark predictor

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
time.sleep(2.0)
class MainWindow(QMainWindow):
    curMAR=0
    curEAR=0

    def viewCam(self):
            # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(image, 0)
        for face in faces:
            shape = self.predictor(image, face)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            self.curMAR=mar
            eyes = shape[rStart:rEnd]
            ear = eye_aspect_ratio(eyes)
            self.curEAR=ear
            #if mar>MOUTH_AR_THRESH:
                #self.ui.mouth.setText("Mouth State:open")
                #self.ui.general.setText("General Status:Yawn")
            #else:
                ##self.ui.mouth.setText("Mouth State:Close")
                #self.ui.general.setText("Mouth State:Awake")    
        # compute the bounding box of the face and draw it on the frame

   
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label_6.setPixmap(QPixmap.fromImage(qImg))
        qImg2 = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.ui.label_20.setPixmap(QPixmap.fromImage(qImg2))
        qImg3 = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(qImg3))
    # start/stop timer

    def controlTimer(self):
        if not self.timer.isActive():
                # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
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
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow2()
        self.ui.setupUi(self)
        self.ui.btn_open_file.clicked.connect(self.Button)
        self.ui.pushButton_2.clicked.connect(self.Button)
        self.ui.btn_open_file.setStyleSheet("border-right: 5px solid rgb(20, 255, 236);")
        self.ui.btn_mouth.clicked.connect(self.Button)
        self.ui.btn_new_user.clicked.connect(self.Button)
        self.ui.btn_settings.clicked.connect(self.Button)
        self.graphWidget = pg.PlotWidget(self)
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.graphWidget.sizePolicy().hasHeightForWidth())
        self.graphWidget.setSizePolicy(sizePolicy)
        self.graphWidget.setMinimumSize(QSize(0, 0))
        self.ui.verticalLayout_14.addWidget(self.graphWidget,0,0,1,1)
        
        self.ui.verticalLayout_14.setMargin(0)
        self.x = list(range(100))  # 100 time points
        self.y = [0 for _ in range(100)]  # 100 data points

        self.graphWidget.setBackground((20,20,20,0))
        self.graphWidget.setTitle("Eye aspect ratio curve", color="#f12345", size="30pt")
        styles = {"color": "#f00", "font-size": "20px"}
        self.graphWidget.setLabel("left", "EAR", **styles)
        self.graphWidget.setLabel("bottom", "Time(fps)", **styles)
        pen = pg.mkPen(color=(25, 255, 236))
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
        
        
        
        
        self.graphWidget2 = pg.PlotWidget(self)
        self.ui.verticalLayout_10.addWidget(self.graphWidget2,0,0,1,1)

        self.x2 = list(range(100))  # 100 time points
        self.y2 = [0 for _ in range(100)]  # 100 data points

        self.graphWidget2.setBackground((20,20,20,0))
        self.graphWidget2.setTitle("Mouth aspect ratio curve", color="#f12345", size="30pt")
        styles2 = {"color": "#f00", "font-size": "20px"}
        self.graphWidget2.setLabel("left", "MAR", **styles2)
        self.graphWidget2.setLabel("bottom", "Time(fps)", **styles2)
        pen2 = pg.mkPen(color=(25, 255, 236))
        self.data_line2 =  self.graphWidget2.plot(self.x2, self.y2, pen=pen2)
        # create a timer
        self.timer = QTimer()
        #self.ui.btn_new_user.clicked.connect(eve)
        #self.ui.btn_new_user.installEventFilter(self)
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.timeout.connect(self.update_plot_data2)
        QTimer.singleShot(2,self.controlTimer)
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
        UIFunctions.addNewMenu(self, "EYE", "btn_new_user", "url(:/16x16/icons/16x16/cil-eye-open.png)", True)
        UIFunctions.addNewMenu(self, "Mouth", "btn_mouth", "url(:/16x16/icons/16x16/cil-mouth.png)", True)
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
        #self.show()
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
            if self.ui.btn_mouth.isEnabled():
                self.ui.btn_mouth.setEnabled(False)
            else:
                self.ui.btn_mouth.setEnabled(True)
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
    
