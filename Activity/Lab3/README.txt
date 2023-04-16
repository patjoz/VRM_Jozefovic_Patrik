Dear user of planar 3arm robot SCARA,

first of all, I would suggest to briefly read comments in the code itself, secondly, as a brief instruction, the program is controlled from the main_3R.py

there, you can adjust values in lines reagarding to the kinematics *test_kin*. 
***************************** FORWARD KINEMATICS *****************************
As of forward kinematics calculation 'FK': you can adjust values in brackets [45,45,45]
these values correspond with the angles of single joints [angle of J1, angle of J2, angle of J3].

***************************** INVERSE KINEMATICS *****************************
In case of inverse kinematics 'IK': you are choosing point, where end-effector should end [x,y] as well as angle of enf-effector in the point, relative to base i.e. angle is calculated from x-axis counter-clockwise.
Last adjustable parameter is configuration in which the robot reaches the desired point, either 0 or 1 are acceptable.
Final command should look like scara.inverse.kinematics.([0.5, 0.5], 90, 0)
Meaning - desired point is x = 0.5, y = 0.5, end-effector angle = 90° and with default configuration cfg = 0. If this point is not reachable, WARNING will 
appear. If point is reachable but with different end-effector angle, an alternative angle is calculated and applied, WARNING will also appear notifying the user.

***************************** PYTHON LIBRARIES *****************************
NOTE: All needed python libraries are contained in requirements.txt file, uploaded with program.
****************************************************************************

Thank you for understanding and have fun!
