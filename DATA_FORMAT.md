DATA FORMATS


DATA FORMAT of files outputted from Unity
- cam_pos_TIMESTAMP.txt:
    [ x, y, z, yaw, pitch, roll ]

    - angles are represented in Degrees

    Coordiante system assumptions: 
    - Left-handed, Y-up
    - Postive yaw change corresponds with clockwise rotation
    - When camera is facing 0 deg (ie, yaw = 0):
        - X change is +rightward/-leftward
        - Y is vertical axis (is constant for our custom dataset) 
        - Z change is +forward/-backward

- raw image types available (By string):
    - "_img", "_depth", "_flow", "_id", "_layer", "_normals"


DATA FORMAT THAT OUR MODEL EXPECTS
    * this is the format the model expects to be returned from the dataloader

    [ X, Y, Z, cos(yaw), sin(yaw), cos(pitch), sin(pitch), cos(roll), sin(roll) ]

    Coordiante system assumptions:
    - Right-handed, Z-up
    - by default, we assume euler angles are ordered as yaw, pitch, roll, respectively, where:
        - Postive yaw change corresponds with counter-clockwise rotation
        - pitch corresponds to looking up/down
        - roll corresponds to tilting head
    - When camera is facing 0 deg (ie, yaw = 0):
        - X change is +rightward/-leftward
        - Y change is +forward/-backward
        - Z is vertical axis (current model expects Z to be constant)
    
    Other data assumptions:
    - Quaternions should be in XYZW format


Conclusions on MOVI data formatting (as they are formatted after saved in our pickle format):
    -   NOTE: The MOVI data iteself saves rotation onformation as quaternions. During the conversion to our datamap pickle format, we convert the quaternion to euler angles -- at this step, as of now, we choose an intrinsic ZXY representation for the Euler angles, and then therefore built the rest of our model code to also assume an intrinsic ZXY representation 
    -   A right-handed system with upward axis +Z, forward axis +Y, and rightward axis +X:
        ⁃	angle[0] is rotate around Z-axis (we may refer to as 'yaw'), 
        -   angle[1] is rotate around X-axis (we may refeer to as 'pitch')
        -   angle[2] is rotate around Y-axis (should always be 0 for movi-e)
        -   when the set of angles is [0,0,0], the camera is 'looking' down along the negative Z-axis (so, in this default orientation, the +Y-axis can be thought of as protruding out of the top  of the camera's head)
