import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
import cv2.aruco as aruco
from numpy import load
import numpy as np
import math


#################### FUNCTION TO ENABLE BASIC FUNCTIONS #############################

def init_gl():
    glEnable(GL_DEPTH_TEST)                               #TO ENABLE DEPTH TESTING,SO THAT HIDDEN SURFACES ARE REMOVED
    glMatrixMode(GL_PROJECTION)                           #TO ENABLE PROJECTION MATRIX TO DEFINE A CLIPPING SPACE(THE SHAPE IS THAT OF A FRUSTUM)
    glLoadIdentity()                                      #TO LOAD IDENTITY MATRIX,THE PROJECTION MATRIX RESETS TO IDENTITY MATRIX SO THAT NEW PARAMETERS ARE NOT COMBINED WITH PREVIOUS ONES
    gluPerspective(45, 640.0 /480, 0.1, 1000)             #PERSPECTIVE VIEW GIVES US A VIEW IN WHICH FARTHER OBJECTS APPEARS SMALLER COMPARED TO NEARER OBJECTS
    glViewport(0, 0, 640, 480)                            #IT TELLS WHICH PART OF WINDOW WILL BE VISIBLE,ITS UNIT IS SCREEN PIXEL UNITS
    global texture_object, texture_background             #DEFINING NAME OF TEXTURES TO BE USED FOR TEAPOT AND BACKGROUND RESPECTIVELY
    texture_object = glGenTextures(1)                     # RETURNS NAME OF THE TEXTURE WHICH IS STORED IN TEXTURE_OBJECT,1 DENOTE THE NO. OF TEXTURE NAME RETURNED
    texture_background=glGenTextures(1)                   # RETURNS NAME OF THE TEXTURE WHICH IS STORED IN TEXTURE_BACKGROUND,1 DENOTE THE NO. OF TEXTURE NAME RETURNED
    glEnable(GL_TEXTURE_2D)                               #TO ENABLE 2D TEXTURE MODE
    glEnable(GL_LIGHTING)                                 #TO ENABLE LIGHTING
    glEnable(GL_LIGHT0)                                   #TO USE LIGHT 0
    glActiveTexture(GL_TEXTURE0)                          #THE ACTIVE TEXTURE UNIT, BY DEFAULT IT IS THE ACTIVE TEXTURE UNIT


####################### FUNCTION TO CHECK WHETHER ANY ARUCO MARKER HAS BEEN DETECTED IN THE IMAGE OR NOT #######################

def check_markers(img):
    ar_module=aruco.Dictionary_get(aruco.DICT_5X5_250)                      #TO SELECT AN ARUCO DICTIONARY
    detect=aruco.DetectorParameters_create()                                #INITIALIZING THE DETECTOR
    mc,mid,_=aruco.detectMarkers(img,ar_module,parameters=detect)           #TO GET ID AND CORNER POINTS OF DETECTED ARUCO MARKERS
    return mc                                                               #RETURNING CORNER POINTS AS A MEANS TO CHECK WHETHER ANY MARKER IS DETECTED OR NOT


###################### FUNCTION TO GET THE LIST OF ARUCO MARKERS DETECTED IN THE IMAGE #########################
###################### THE LIST CONTAINS ARUCO ID,COORDINATE OF ITS CENTRE,ROTATION VECTOR OF ITS CENTRE,TRANSLATION VECTOR OF ITS CENTRE,CORNER POINTS OF MARKER ##############

def detect_markers(img,mtx,dist):
    ar_module=aruco.Dictionary_get(aruco.DICT_5X5_250)                       #TO SELECT AN ARUCO DICTIONARY
    detect=aruco.DetectorParameters_create()                                 #INITIALIZING THE DETECTOR
    mc,mid,_=aruco.detectMarkers(img,ar_module,parameters=detect)            #TO GET ID AND CORNER POINTS OF DETECTED ARUCO MARKERS
    aruco_lst=[]                                                             #LIST TO ADD DETECTED ARUCO MARKERS' PROPERTIES

    ################### LOOP TO ADD PROPERTIES OF DETECTED MARKERS IN THE LIST ###################################

    for i in range(len(mid)):
        aruco_id=mid[i]                                                            #ARUCO ID IS STORED IN ARUCO_ID
        rvec,tvec,_=aruco.estimatePoseSingleMarkers(mc,100,mtx,dist)               #ROTATION AND TRANSLATION VECTORS OF MARKER'S CENTRE IS CALCULATED AND STORED IN RVEC AND TVEC
        x=(mc[i][0][0][0]+mc[i][0][1][0]+mc[i][0][2][0]+mc[i][0][3][0])/4          # X-COORDINATE OF MID POINT OF ARUCO IS FOUND
        y=(mc[i][0][0][1] + mc[i][0][1][1] + mc[i][0][2][1] + mc[i][0][3][1])/4    # Y-COORDINATE OF MID POINT OF ARUCO IS FOUND
        centre=(x,y)                                                               # CENTRE COORDINATES IS STORED IN A TUPLE
        tpl=(np.array([aruco_id]),centre,rvec[i],tvec[i],mc[i])                    #ALL THE PROPERTIES OF A DETECTED ARUCO IS COLLECTED AND STORED IN A TUPLE
        aruco_lst.append(tpl)                                                      #THE CREATED TUPLE IS ADDED IN THE LIST
    return aruco_lst                                                               #THE FINAL LIST IS RETURNED


####################### FUNCTION TO DRAW LINES AROUND THE ARUCO MARKERS ##############################
####################### THE ID OF THE MARKER WILL ALSO BE WRITTEN ON TOP OF IT #######################

def show_detected_markers(img,mtx,dist,aruco_lst):
    for i in range(len(aruco_lst)):
        id=aruco_lst[i][0][0][0]                                                                                 #GETTING ID OF ARUCO
        corner1=tuple(aruco_lst[i][4][0][0])
        corner2 = tuple(aruco_lst[i][4][0][1])
        corner3 = tuple(aruco_lst[i][4][0][2])
        corner4 = tuple(aruco_lst[i][4][0][3])                                                                   #GETTING COORDINATES OF ALL THE FOUR CORNER POINTS OF THE DETECTED ARUCO
        pts=np.float32([[0,0,0]])
        rvec = aruco_lst[i][2]
        tvec = aruco_lst[i][3]
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, mtx, dist)
        centre=tuple(imgpts[0][0])                                                                               #TO GET COORDINATE OF MID POINT OF MARKER
        img = cv2.line(img, corner1, corner2, (0, 0, 255), 5)
        img = cv2.line(img, corner2, corner3, (0, 0, 255), 5)
        img = cv2.line(img, corner3, corner4, (0, 0, 255), 5)
        img = cv2.line(img, corner4, corner1, (0, 0, 255), 5)                                                     #TO DRAW A BOX AROUND THE MARKER
        img=cv2.putText(img,"id="+str(id),centre,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3,cv2.LINE_AA)              #TO WRITE ID OF THE ARUCO
    return img                                                                                                    #RETURNING THE IMAGE WITH DRAWN BORDER AND ID

############################# FUNCTION TO CREATE TEXTURE FOR TEAPOT ######################################

def init_object_texture(text):
    glBindTexture(GL_TEXTURE_2D,texture_object)                                      #BINDING TEXTURE OBJECT texture_object TO TEXTURE UNIT 0
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP)                        #IT SPECIFIES HOW TO WRAP TEXTURE ALONG X-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF X-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)                      #IT SPECIFIES HOW TO WRAP TEXTURE ALONG Y-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF Y-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)                 #APPLYING TEXTURE FILTER
    img=cv2.imread(text)                                                             #CONVERTING IMAGE IN ARRAY FORM (BGR FORMAT)

    img = Image.fromarray(img)                                                       #CONVERTING IMAGE FROM ARRAY FORM TO PILLOW IMAGE FORM
    width=img.size[0]                                                                #CALCULATING WIDTH OF IMAGE
    height=img.size[1]                                                               #CALCULATING HEIGHT OF IMAGE
    img=img.tobytes("raw", "BGRX")                                                   #CONVERTING IMAGE TO BYTES

    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,img)#CREATES A 2-D TEXTURE IMAGE


############################## FUNCTION TO DISPLAY TEAPOT ################################################

def overlay(aruco_lst):
    error=0
    rvecs=aruco_lst[0][2]
    tvecs=aruco_lst[0][3][0]
    rmtx = cv2.Rodrigues(rvecs)[0]                                            #CONVERTING ROTATION VECTOR INTO ROTATION MATRIX

    ############### OPENGL AND OPENCV HAVE SAME MAGNITUDE OF ROTATION MATRIX ###################
    ############### THEY HAVE DIFFERENT TRANSLATION VECTORS ####################################

    K=101.13058796569                                                          #K IS THE SCALING FACTOR FOR Z-TRANSLATION VECTOR

    #  K = (FARTHEST VALUE OF Z-TRANSLATION VECTOR DETECTED BY CAMERA IN OPENCV SCREEN) - (CLOSEST VALUE OF Z-TRANSLATION VECTOR DETECTD BY CAMERA IN OPENGL SCREEN)/
    #       (58 WHICH IS THE VALUE BY WHICH BACKGROUND HAS BEEN TRANSLATED ALONG -VE Z-AXIZ) - (0.1 WHICH IS OUR NEAR CLIPPING PLANE)


    z=tvecs[2]/K     # Z-TRANSLATION VECTOR OF OPENGL HAS BEEN CREATED BY USING SCALING FACTOR

    ############### THE MAGNITUDES OF TRANSLATION VECTORS ARE DIFFERENT IN OPENCV AND OPENGL #####################
    ############### BUT THE ANGLES BETWEEN THEM ARE SAME IN BOTH CASES ###########################################

    thetax=math.atan(abs(tvecs[2]/tvecs[0]))                                    # TAN(THETAX) = Z-TRANSLATION VECTOR/X-TRANSLATION VECTOR    IN OPENCV
    thetay=math.atan(abs(tvecs[2]/tvecs[1]))                                    # TAN(THETAY) = Z-TRANSLATION VECTOR/Y-TRANSLATION VECTOR    IN OPENCV

    abs(thetax)
    abs(thetay)                                                                 # FOR NOW, ONLY POSITIVE ANGLES ARE NEEDED
    if(z>10 and z<22):
        error=2.5
    if(z>=22):
        error=5
    x=(z-error)/(math.tan(thetax))
    y=(z-error)/(math.tan(thetay))                                                      # FINDIND X AND Y TRANSLATION VECTORS OF OPENGL FORMAT BY USING PREVIOUSLY FOUND ANGLES

    ################ BY USING ABOVE METHOD WE WILL BE ONLY GETTING POSITIVE X AND Y TRANSLATION VECTORS ##############
    ################ SO THE ALGORITHM USED BELOW WILL CONVERT IT INTO NEGATIVE FORM WHENEVER IT IS REQUIRED ##########

    if(tvecs[0]<0):
        x=-x
    if(tvecs[1]<0):
        y=-y


    view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], x],             #VIEW MATRIX OF OPENCV FORMAT
                            [rmtx[1][0], rmtx[1][1], rmtx[1][2], y],
                            [rmtx[2][0], rmtx[2][1], rmtx[2][2], z],
                            [0.0, 0.0, 0.0, 1.0]])

    inverse_matrix = np.array([[1.0, 1.0, 1.0, 1.0],                             #AS DIRECTION OF Y AND Z AXIS ARE OPPOSITE IN OPENCV AND OPENGL REVERSE THEM BY MULTIPLYING THEM BY -1
                               [-1.0, -1.0, -1.0, -1.0],                         #X AXIS HAS SAME DIRECTION IN BOTH FORMATS
                               [-1.0, -1.0, -1.0, -1.0],
                               [1.0, 1.0, 1.0, 1.0]])

    view_matrix = view_matrix * inverse_matrix
    view_matrix = np.transpose(view_matrix)                                      #CONVERTING VIEW MATRIX FROM ROW MAJOR FORMAT(USED IN OPENCV) TO COLUMN MAJOR FORMAT(USED IN OPENGL)


    ######################### LOADING AND USING VIEW MATRIX FOR TEAPOT ###########################

    glPushMatrix()                                                               #COPYING THE TOPMOST MATRIX OF MODELVIEW STACK WHICH IS IDENTITY MATRIX AND BECOMING ITSELF THE TOPMOST MATRIX
    glLoadMatrixd(view_matrix)                                                   #MULTIPLYING THE IDENTITY MATRIX WITH OUR VIEW MATRIX
    glutSolidTeapot(1)                                                           #FUNCTION TO CREATE TEAPOT
    glPopMatrix()                                                                #DELETING THE TOPMOST MATRIX OF MODELVIEW STACK


############################# FUNCTION TO DRAW BACKGROUND IN OPENGL ################################################

def draw_background(frame):
    glBindTexture(GL_TEXTURE_2D, texture_background)                              #BINDING TEXTURE OBJECT texture_background TO TEXTURE UNIT 0
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    GL_CLAMP)                                                     #IT SPECIFIES HOW TO WRAP TEXTURE ALONG X-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF X-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    GL_CLAMP)                                                     #IT SPECIFIES HOW TO WRAP TEXTURE ALONG Y-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF Y-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)              #APPLYING TEXTURE FILTER



    frame = Image.fromarray(frame)                                                 # CONVERTING IMAGE FROM ARRAY FORM TO PILLOW IMAGE FORM
    width = frame.size[0]                                                          # CALCULATING WIDTH OF IMAGE
    height = frame.size[1]                                                         # CALCULATING HEIGHT OF IMAGE
    frame = frame.tobytes("raw", "BGRX")                                           # CONVERTING IMAGE TO BYTES

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame) #CREATES A 2-D TEXTURE IMAGE

    ############### CREATING A RECTANGLE AND APPLYING TEXTURE ON IT,THE RATIO OF RECTANGLE IS MAINTAINED AT 4:3 ########

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0);glVertex2f(-32.0, 24.0)
    glTexCoord2f(0.0, 1.0);glVertex2f(-32.0, -24.0)
    glTexCoord2f(1.0, 1.0);glVertex2f(32.0, -24.0)
    glTexCoord2f(1.0, 0.0);glVertex2f(32.0, 24.0)
    glEnd()




############################ THIS FUNCTION RUNS CONTINOUSLY TO DISPLAY THE SCENE IN OPENGL ############################

def drawGLScene():
    glClearColor(0.0, 0.0, 0.0, 0.0)                               #TO SELECT COLOUR WHICH WILL BE DISPLAYED AFTER SCREEN IS CLEARED
    glClearDepth(1.0)                                              #TO SELECT VALUE OF DEPTH/Z BUFFER AFTER SCREEN IS CLEARED,1.0 MEANS THE FARTHEST
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)             #CLEARING COLOUR BUFFER AND DEPTH BUFFER


    ret, frame = cap.read()                                        #GETTING THE FRAME READ BY CAMERA



    if ret == True:
        mc = check_markers(frame)                                   # FUNCTION TO CHECK WHETHER A MARKER IS PRESENT IN THE IMAGE
        glPushMatrix()                                              # COPYING THE TOP MATRIX IN PROJECTION MATRIX STACK AND BECOMING ITSELF THE TOP MATRIX
        glTranslate(0,0,-58)                                        # TRANSLATING OUR BACKGROUND TO Z=-58 TO FIT ON THE SCREEN AND GIVING US A LARGER DEPTH OUR TEAPOT
        draw_background(frame)                                      # APPLYING THE TEXTURE ON OUR RECTANGLE
        glPopMatrix()                                               # DESTROYING THE TOP MATRIX OF PROJECTION MATRIX STACK



        ##################### IF ANY MARKER IS DETECTED IN THE IMAGE THEN BELOW ALGORITHM IS EXECUTED #################

        if (len(mc) != 0):

            aruco_lst = detect_markers(frame, mtx, dist)             # FUNCTION TO RETURN A LIST OF ARUCO MARKERS AND THEIR PROPERTIES
            img = show_detected_markers(frame, mtx, dist, aruco_lst) # FUNCTION TO DRAW BORDER AROUND DETECTED MARKERS AND TO DISPLAY ITS ID
            glMatrixMode(GL_MODELVIEW)                               #LOADING MODELVIEW STACK
            glLoadIdentity()                                         #LOADING IDENTITY MATRIX

            ################ WHEN ARUCO ID=2 IS DETECTED ##################

            if(aruco_lst[0][0][0][0]==2):
                text=r"E:\ROBOLUTION SUMMER PROJECT\Task 1.2\Problem Statement\texture_1.png"
                init_object_texture(text)                            # FUNCTION TO CREATE TEXTURE FOR TEAPOT
                overlay(aruco_lst)                                   # FUNCTION TO DISPLAY TEAPOT


            ################ WHEN ARUCO ID=8 IS DETECTED ##################

            if (aruco_lst[0][0][0][0] == 8):
                text = r"E:\ROBOLUTION SUMMER PROJECT\Task 1.2\Problem Statement\texture_4.png"
                init_object_texture(text)                            # FUNCTION TO CREATE TEXTURE FOR TEAPOT
                overlay(aruco_lst)                                   # FUNCTION TO DISPLAY TEAPOT



            cv2.imshow("d", img)                                     #TO SHOW IMAGE WITH BORDERED ARUCO ON OPENCV SCREEN

        ############# IF NO MARKER IS DETECTED THEN ONLY BACKGROUND IS DISPLAYED IN OPENGL AND OPENCV
        
        else:
            cv2.imshow("d", frame)                                   #TO SHOW IMAGE WITHOUT BORDERED ARUCO ON OPENCV SCREEN


        cv2.waitKey(1)                                               #GAP BETWEEN TWO FRAMES OF OPENCV IS 1 MILLI SECOND

    glutSwapBuffers()                                                #SWAPS THE FRONT AND BACK BUFFER







data=load(r"E:\ROBOLUTION SUMMER PROJECT\Task 0.2\Camera.npz")       # LOADING THE FILE CONTAINING CAMERA PARAMETERS
dist=data["dist"]                                                    # TAKING DATA IN FILE NAME DIST
mtx=data["mtx"]                                                      # TAKING DATA IN FILE NAME MTX
cap=cv2.VideoCapture(0)                                              #TO OPEN OUR CAMERA
glutInit()                                                           #CREATES A GLUT OBJECT THAT ALLOWS US TO CUSTOMIZE OUR WINDOW
glutInitWindowSize(640, 480)                                         #WINDOW SIZE
glutInitWindowPosition(100, 100)                                     # WINDOW DISPLAY POSITION
glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH|GLUT_DOUBLE)              # DISPLAY COLOUR OF RGBA FORMAT,INCLUDING DEPTH BUFFER AND DOUBLE BUFFER IS ALSO SELECTED
glutCreateWindow("hello world")                                      #NAME OF WINDOW
init_gl()                                                            #FUNCTION TO INITIALISE BASIC FUNCTIONS
glutDisplayFunc(drawGLScene)                                         #IT RUNS THE FUNCTION DRAWGLSCENE WHEN WINDOW IS DISPLAYED
glutIdleFunc(drawGLScene)                                            #IT CONTINOUSLY RUNS THE DRAWGLSCENE FUNCTION
glutMainLoop()                                                       #THIS FUNCTION LOOKS AT THE EVENTS IN THE QUEUE,FOR EACH EVENT IN THE QUEUE IT EXECUTES A CALL BACK FUNCTION
                                                                     #IF NO CALL BACK FUNCTION IS DEFINED FOR THAT EVENT THEN THAT EVENT IS IGNORED
