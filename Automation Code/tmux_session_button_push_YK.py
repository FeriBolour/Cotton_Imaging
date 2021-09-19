import libtmux
import os
import time
import RPi.GPIO as GPIO

#GPIO.cleanup() # Always cleans up the setup pins if previously setup (needed to reset the pin value (input value) to 1)

os.system("gnome-terminal -- /bin/sh -c 'tmux new-session -s ros -n New'")
server = libtmux.Server()
#print(server.list_sessions())
session = server.get_by_id('$0')
#print(session)
window = session.new_window(attach=False, window_name="Cotton_img")
session.kill_window("New")
#session.kill_window("Cotton_img")
#window = session.attached_window()
#print(window)
#pane1 = window
window.split_window(attach=False, vertical=False)
#pane3 = window.split_window(attach=False)

#print(pane1)
#print(pane2)

#pane2.select_pane()
pane0 = window.select_pane('0')

#pane0.send_keys('roslaunch realsense2_camera opensource_tracking.launch', enter=True)
#pane0.enter()

pane1 = window.select_pane('1')

button_input_new = 0 # initial value expected to be true when button is connected
flag = True

GPIO.setmode(GPIO.BOARD) # set the GPIO board
GPIO.setup(15,GPIO.IN) # setup pin 15 as input



while True:
    button_input_old = GPIO.input(15) # This will be our button input that will constantly have the same value depending on button push (0 or 1)
      
    #if button_input_new != 1:
    #    if flag == True:
    #        pane1.send_keys('echo nice!', enter=True)
    #        pane0.send_keys('echo good job!', enter=True)
    #        flag = False
    #    else:
    #        pane1.send_keys('C-c', enter=False, suppress_history=False)
    #        pane0.send_keys('C-c', enter=False, suppress_history=False)
    #        flag = True
    #    time.sleep(2)
    

    if button_input_new == 1 and button_input_old == 0:
        pane1.send_keys('echo nice!', enter=True)
        pane0.send_keys('echo good job!', enter=True)
        button_input_new = 2
        #button_input_old = GPIO.input(15)
        #time.sleep(2)
    elif button_input_new == 2 and button_input_old == 0:
        pane1.send_keys('C-c', enter=False, suppress_history=False)
        pane0.send_keys('C-c', enter=False, suppress_history=False)
        button_input_new = 1
    elif button_input_new == 0 and button_input_old == 1:
        time.sleep(2)
        pane1.send_keys('stty  -echo; echo Hi ;stty  echo')
        pane0.send_keys("(echo 'Welcome to Cotton Scanner...Please press the button to start scanning...' &)")
        button_input_new = 1
        continue   
    #else:
        #print('Invalid input')
        #continue
    time.sleep(0.5)    

     
#pane1.enter()
#time.sleep(30)
#pane0.select_pane()
#pane0.send_keys('C-c', enter=False, suppress_history=False)

# GPIO.cleanup() # Always cleans up the setup pins if previously setup (needed to reset the pin value (input value) to 1)

print(window.list_panes())
#session.kill_window("Cotton_img")
#pane1.select_pane()
