Hi all,

My goal was to make accessing the robots easier this semester but there are unavoidable issues at the moment.

Currently I have the robot named Bomber working for Teleop. I will get at least 5 for you all to use by Monday. For now the TOP SHELF OF THE GREEN BOOKSHELF will be reserved for functional robots only. Please do not place any other robots their or working robots anywhere else.

In the back of the room there is a circular table set up with a monitor, keyboard, and mouse. Plug these into Bomber to access the robot's Raspberry Pi. When you turn on the robot it should boot into a terminal (without need for a password - the password is burger for all of the robots). To start the robot's desktop enter -

``startx``

This should bring you to the desktop of the robot. In the upper right hand corner will be the WiFi signal. Click that and select eduroam. 

A window prompt will appear. You should choose --

``
Wi-Fi security - WPA & WPA2 Enterprise
Autentication - Protected EAP (PEAP)
Anonymous Identity - leave blank
CA certificate - (None)
      select No CA certificate is required
PEAP version - Automatic
Inner authentication - MSCHAPv2
Username - <your GaTech Username>@gatech.edu
Password - <your GaTech Password>
``

Your username and password is saved in plain text for anyone to see. Thus, I cannot have these configured with my credentials as you all could see them, and, if you were a benevolent mischief maker, log into canvas give everyone As and never come to class =(. Make sure to delete this connection configuration on the robot when you are done if you care about your password to your gatech account being visible.

From here you can determine the IP address of the robot through the terminal with the ifconfig command. You can edit the bash scripts so your PC is the rosmaster and the robot's IP is correct.
