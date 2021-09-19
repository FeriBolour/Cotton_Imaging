import libtmux
import os
import keyboard
import time

#GPIO.cleanup() # Always cleans up the setup pins if previously setup (needed to reset the pin value (input value) to 1)

# ALL OF THIS BELOW NEEDS TO BE MADE INTO A FUNCTION to REMOVE CONFUSION!!!

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

# MAKE INTO A FUNCTION UP TO THIS POINT!!!

# Track File names and numbers and save .db and .bag files in different folders
count = 0 # count is NOT REQUIRED!!!
myfilename = str(input('Enter the name of the files: ')) # This needs to be asked at every restart (space push button)!!! WRITE WITHIN THE WHILE LOOP!
database_path = 'somepath' # This needs to be defaulted to a path to avoid retyping every time!!! FOR SAVING + FOR LAUNCH FILES!!!

# THIS IS A GOOD IDEA TO KEEP TRACK OF ALL THE FILES BEING SCANNED!!! OR IT COULD BE POSTPROCESSED (IF TRUSTED TO BE NOT FORGOTTEN LATER)
# Create New Main Folder
# Create 2 sub folders inside Main Folder (for .db and .bag)

# Camera Parameter change Commands
change_param = 'rosrun dynamic_reconfigure dynparam set /camera/rgb_camera '
enable_auto_exposure = 'enable_auto_exposure '
auto_exposure_input = 'False' # default (can be changed to True if desired) # Need adjustments!!! (give the ability to use autoexposure --> True or False)
exposure_input= str(input('Enter the desired camera exposure level: ')) # Need adjustments!!! If 'True' don't let the user give input
exposure_level = 'exposure ' + exposure_input # Need adjustments!!! skip this if 'True'
print(exposure_level)

# Commands to save Rosbag file
rosbag_record = 'rosbag record -o '
rosbag_data = '.bag /camera/aligned_depth_to_color/image_raw /camera/color/image_raw /tf_static /tf'

# Commands to stop the process
    # ctrl + C

# Start Launch File
print('Press space to continue...')
print('Press enter to exit the code...')
while True:

    # NEED TO ASK THE NAME OF THE .db file at every restart (space button push)!!!

    count = count + 1 # NOT REQUIRED!
    newfilename = myfilename + '_' + str(count) # NOT REQUIRED! ASK FOR USER INPUT EVERYTIME!!!
    print(newfilename)

    # WAITS FOR AN INPUT FROM KEYBOARD EVERYTIME!!!
    input_3 = keyboard.read_key()
    # If ESC BUTTON IS PUSHED EXIT THE CODE
    if input_3 == 'esc':
        pane1.send_keys('C-b',enter=False, suppress_history=False)
        pane1.send_keys('x', enter=True)
        pane1.send_keys('y', enter=True)
        pane0.send_keys('C-b',enter=False, suppress_history=False)
        pane0.send_keys('x', enter=True)
        pane0.send_keys('y', enter=True)
        print('Exiting Code!')
        # SAVE METADATAFILE BEFORE EXITING THE CODE!!!
        break
    # IF SPACE BUTTON IS PUSHED START THE SCAN
    elif input_3 == 'space':

        # Run ROS LAUNCH (Terminal # 1)
        print('roslaunch realsense2_camera opensource_tracking.launch')
        pane1.send_keys('roslaunch realsense2_camera opensource_tracking.launch', enter=True) # COMMAND TO START THE SCANNING SESSION
        time.sleep(7)  # Wait 7 seconds to give ROS Launch enough time to start

        # Turn off Auto Exposure and set exposure level (Terminal # 2)
        if count == 1:  # If you want to execute it everytime, remove this if line (maybe keep this for setting autoexpose only once? still does not require count update but, needs a default count value == 1)
            set_auto_exposure = change_param + enable_auto_exposure + auto_exposure_input  # Turn off auto exposure (give more options with if and elif for autoexposure enabled vs disabled)
            set_exposure_level = change_param + exposure_level  # Set exposure level
            print(set_auto_exposure) # NOT REQUIRED?
            print(set_exposure_level) # NOT REQUIRED?
            pane0.send_keys(set_auto_exposure, enter=True)
            time.sleep(1)
            pane0.send_keys(set_exposure_level, enter=True)
        # Record Rosbag file
        record_rosbagfile = rosbag_record + newfilename + rosbag_data # Command to start recording data into a rosbag file
        pane0.send_keys(record_rosbagfile, enter=True) # START SAVING THE ROSBAG FILE!!!

        # WAIT FOR A KEYBOARD PUSH (SPACE)
        keyboard.wait('space')
        # GET THE KEYBOARD VARIABLE NAME TO GIVE COMMANDS => if input_2 = 'space' STOP BOTH TERMINAL-1 AND TERMINAL-2 PROCESSES
        input_2 = keyboard.get_hotkey_name()
        if input_2 == 'space':
            # Stop Process in Terminal # 1
            pane1.send_keys('C-c', enter=False, suppress_history=False)
            time.sleep(3)
            # Stop Process in Terminal # 2
            print('Hello!') # NOT REQUIRED!
            pane0.send_keys('C-c', enter=False, suppress_history=False)
            time.sleep(1)
    # IF ANY BUTTON OTHER THAN 'SPACE' OR 'ESC' IS PUSHED, PRINT A WARNING!!! IF BUTTONS ARE PUSHED BY MISTAKE DURING THE SCANNING SESSION!!!
    else:
        print('Wrong Key! Try again!')
        time.sleep(1)