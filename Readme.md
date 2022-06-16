_#File structure_
**global setting**:gui_base.py

**server**: gui_server.py

**client 1 - 4** : cameras and tof sensors 

_#Pipline_

1. Client 1-4 collect and process data in own file/pipline and transport the result individualy into server.
2. server will stop accepting connect requests when the connected clients is full (4 clients)
3. Only when all 4 input-data (image and tof singal) have reached the server then server will update the GUI display.
4. In recording mode the pressure will only update when the compress fore more than 5 N (OCR buffer is setted as 20 frames).
5. If the pressure is less than 5 N during the recording process, the measurement process on this patient is considered to be over and the previous data is recorded in the corresponding file.
6. Exit btn will kill all processes and threads immediately , if programm is running in recording stage , videos will be saved for sure.
