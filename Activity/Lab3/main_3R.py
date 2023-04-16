"""
## =========================================================================== ## 
MIT License
Copyright (c) 2020 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: main.py
## =========================================================================== ## 
"""

# System (Default Lib.)
import sys
# Own library for robot control (kinematics), visualization, etc. (See manipulator.py)
import manipulator_3R

def main():
    # Initial Parameters -> ABB IRB910SC 
    # Product Manual: https://search.abb.com/library/Download.aspx?DocumentID=3HAC056431-001&LanguageCode=en&DocumentPartId=&Action=Launch

    # Working range (Axis 1, Axis 2, Axis 3)
    axis_wr = [[-140.0, 140.0],[-150.0, 150.0],[-140.0, 140.0]]
    # Length of Arms (Link 1, Link2, Link3)
    arm_length = [0.3, 0.25, 0.2]

    # DH (Denavit-Hartenberg) parameters
    theta_0 = [0.0, 0.0, 0.0]
    a       = [arm_length[0], arm_length[1], arm_length[2]]
    d       = [0.0, 0.0, 0.0]
    alpha   = [0.0, 0.0, 0.0]

    # Initialization of the Class (Control Manipulator)
    # Input:
    #   (1) Robot name         [String]
    #   (2) DH Parameters      [DH_parameters Structure]
    #   (3) Axis working range [Float Array]
    scara = manipulator_3R.Control('ABB IRB 910SC (SCARA) - extension 3rd arm', manipulator_3R.DH_parameters(theta_0, a, d, alpha), axis_wr)

    # Test Results (Select one of the options -> See below)
    test_kin = 'IK'

    if test_kin == 'FK':
        scara.forward_kinematics([45, 45, 100], True)
    elif test_kin == 'IK':
        # Inverse Kinematics Calculation -> Default Calculation method
        # Input data are in shape inverse_kinematics([p1,p2], theta, cfg)
        # p1 - desired x position, p2 - desired y position, theta - desired angle of end effector, cfg - configuration
        scara.inverse_kinematics([0.3, 0.5], 180, 0)
    elif test_kin == 'BOTH':
        scara.forward_kinematics([0.0, 45.0, 45.0], True)
        scara.inverse_kinematics([0, 0.5], 30, 0)

    # 1. Display the entire environment with the robot and other functions.
    # 2. Display the work envelope (workspace) in the environment (depends on input).
    # Input:
    #  (1) Work Envelop Parameters
    #       a) Visible                   [BOOL]
    #       b) Type (0: Mesh, 1: Points) [INT]
    scara.display_environment([True, 1])

if __name__ == '__main__':
    sys.exit(main())