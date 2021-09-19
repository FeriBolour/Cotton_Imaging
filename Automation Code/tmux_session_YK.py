import libtmux
import os
import time

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

button_input_new = 1 # initial value expected to be true when button is connected

while True:
    button_input_old = int(input('Enter an input: ')) # This will be our button input that will constantly have the same value depending on button push (0 or 1)
    
    if button_input_new == 1 and button_input_old == 1:
        pane1.send_keys('echo nice!', enter=True)
        pane0.send_keys('roslaunch realsense2_camera opensource_tracking.launch', enter=True)
        button_input_new = 0
    elif button_input_new == 0 and button_input_old == 0:
        pane1.send_keys('C-c', enter=False, suppress_history=False)
        pane0.send_keys('C-c', enter=False, suppress_history=False)
        button_input_new = 1
    elif button_input_new == 0 or button_input_new == 1:
        continue   
    else:
        print('Invalid input')
        continue
#pane1.enter()
#time.sleep(30)
#pane0.select_pane()
#pane0.send_keys('C-c', enter=False, suppress_history=False)


print(window.list_panes())
#session.kill_window("Cotton_img")
#pane1.select_pane()
