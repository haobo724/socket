_#File structure_
**global setting**:gui_base.py

**server**: gui_server.py

**client 1 - 4** : cameras and tof sensors 



_#How to start programm_

1. run the start.bat file


_#Pipline_

1. Client 1-4 collect and process data in own file/pipline and transport the result individualy into server.
2. server will stop accepting connect requests when the connected clients is full (4 clients)
3. Only when all 4 input-data (image and tof singal) have reached the server then server will update the GUI display.
4. In recording mode the pressure will only update when the compress fore more than 5 N (OCR buffer is setted as 20 frames).
5. If the pressure is less than 5 N during the recording process, the measurement process on this patient is considered to be finished and the previous data is recorded in the corresponding file.
6. Exit btn will kill all processes and threads immediately , if programm is running in recording stage , videos will be saved for sure.

_#How to select bot rec area (only once after mounting)_

1. run pre_cali_bot.bat
2. press s to make a screenshot from live camera
3. press s to choose the roi with mouse (force) then press space to continue
4. press s to choose the roi with mouse (height) then press space to continue
5. press s to choose the roi with mouse (whole display) then press space to finish


_#How to Generate_look_up_table (only once after mounting)_

1. run the programm normally , make sure the paddle already on suitable positon ,e.g. 150mm
2. Pick Pre-recording option on Gui and click Recording button
3. Let the paddle slowly slide down to the bottom *(Don't stop in the middle of a descent)
4. Click Recording button again to stop the recording
5. shut down the programm
6. run Generate_look_up_table.bat file , the look up table will generate in 30 sec.

