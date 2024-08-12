import numpy as np

def return_contexts():
    """
        All dataseta have some contexts i.e. their classification w.r.t. their shapes, and textures
        for the objects; whereas just the texture for the walls, floor, and ceiling.
    """
    objshapelabels = {"Cube":1, "Cylinder":2, "Cylinder_90z":3,\
                        "Sphere":4, "cone":5, "icosphere":6,\
                        "pyramid":7, "standingTorus":8, "Prism":9,\
                        "bunny":10, "monkey":11}

    objtexlabels = {"Metalpattern02":1, "Metalpattern04":2, "Metalpattern05":3,\
                    "Metalpattern07":4, "Metalpattern08":5, "Metalpattern09":6,\
                    "Metalpattern11":7, "Metalpattern13":8, "Metalpattern15":9,\
                    "Metalpattern18":10, "Metalpattern21":11, "Metalpattern26":12,\
                    "Metalpattern28":13, "blueMat":14, "greenMat":15,\
                    "pinkMat":16, "redMat":17, "yellowMat":18}


    walltexlabels = {"lg_style_05_wall_red_bright_d":1, "lg_style_02_wall_yellow_d":2,\
                    "lg_style_01_wall_purple_d":3, "lg_style_01_wall_light_m":4,\
                    "lg_style_03_wall_light_m":5, "lg_style_02_wall_dblue_d":6,\
                    "lg_style_03_wall_gray_d":7, "lg_style_01_wall_green_bright_d":8,\
                    "lg_style_01_wall_yellow_d":9, "lg_style_04_wall_red_bright_d":10,\
                    "lg_style_01_wall_green_d":11, "lg_style_05_wall_yellow_d":12,\
                    "lg_style_01_wall_red_d":13, "lg_style_03_wall_purple_d":14,\
                    "lg_style_02_wall_yellow_bright_d":15, "lg_style_02_wall_purple_d":16,\
                    "lg_style_02_wall_blue_bright_d":17, "lg_style_02_wall_blue_d":18,\
                     "lg_style_01_wall_cerise_d":19, "lg_style_05_wall_red_d":20,\
                     "lg_style_05_wall_yellow_bright_d":21, "lg_style_04_wall_green_bright_d":22,\
                     "lg_style_04_wall_cerise_d":23, "lg_style_03_wall_gray_bright_d":24,\
                     "lg_style_04_wall_green_d":25, "lg_style_01_wall_blue_d":26,\
                     "lg_style_03_wall_blue_d":27, "lg_style_03_wall_cyan_d":28,\
                     "lg_style_03_wall_orange_bright_d":29, "lg_style_04_wall_red_d":30,\
                     "lg_style_02_wall_lgreen_d":31, "lg_style_01_wall_red_bright_d":32,\
                     "lg_style_04_wall_purple_d":33, "lg_style_03_wall_orange_d":34,\
                     "wall01b":35,"wall02b":36,"wall03b":37,"wall04b":38,"wall05b":39,\
                     "wall06b":40, "wall07b":41,"wall08b":42,"wall09b":43,"wall10b":44,\
                     "wall11b":45,"wall12b":46, "wall13b":47,"wall14b":48,"wall15b":49,\
                     "wall16b":50,"wall17b":51,"wall18b":52}


    floortexlabels = {"lg_style_04_floor_orange_d":1,\
                    "lg_style_02_floor_blue_bright_d":2,\
                    "lg_style_05_floor_light2_m":3,\
                    "lg_style_03_floor_red_d":4,\
                    "lg_style_01_floor_blue_team_d":5,\
                    "lg_style_04_floor_blue_d":6,\
                    "lg_style_04_floor_dorange_d":7,\
                    "lg_style_03_floor_green_bright_d":8,\
                    "lg_style_05_floor_light1_m":9,\
                    "lg_style_05_floor_orange_d":10,\
                    "lg_style_02_floor_light_m":11,\
                    "lg_style_02_floor_green_d":12,\
                    "lg_style_05_floor_orange_bright_d":13,\
                    "lg_style_01_floor_blue_bright_d":14,\
                    "lg_style_03_floor_blue_bright_d":15,\
                    "lg_style_02_floor_blue_d":16,\
                    "lg_style_03_floor_light3_m":17,\
                    "lg_style_02_floor_orange_d":18,\
                    "lg_style_03_floor_blue_d":19,\
                    "lg_style_03_floor_light1_m":20,\
                    "lg_style_04_floor_blue_bright_d":21,\
                    "lg_style_03_floor_purple_d":22,\
                    "lg_style_01_floor_orange_d":23,\
                    "lg_style_03_floor_light2_m":24,\
                    "lg_style_04_floor_orange_bright_d":25,\
                    "lg_style_03_floor_green_d":26,\
                    "lg_style_02_floor_green_bright_d":27,\
                    "lg_style_01_floor_light_m":28,\
                    "lg_style_05_floor_blue_bright_d":29,\
                    "lg_style_01_floor_red_team_d":30,\
                    "lg_style_01_floor_blue_d":31,\
                    "lg_style_05_floor_blue_d":32,\
                    "lg_style_04_floor_cyan_d":33,\
                    "lg_style_01_floor_orange_bright_d":34,\
                    "lg_style_03_floor_orange_d":35,\
                    "Woodenfloor02":36,\
                    "Woodenfloor03":37,\
                    "Woodenfloor04":38,\
                    "Woodenfloor05":39,\
                    "Woodenfloor06":40,\
                    "Woodenfloor07":41,\
                    "Woodenfloor08":42,\
                    "Woodenfloor09":43,\
                    "Woodenfloor10":44,\
                    "Woodenfloordiffuse":45}

    ceilingtexlabels = {"Office_Ceiling_v1_(Standard)":1,\
                        "Office_Ceiling_v2_(Standard)":2,\
                        "Office_Ceiling_v3_(Standard)":3,\
                        "Office_Ceiling_v4_(Standard)":4,\
                        "Office_Ceiling_v5_(Standard)":5}

    contexts = {"objshape":objshapelabels,\
                "objtex":objtexlabels,\
                "walltex":walltexlabels,\
                "floortex":floortexlabels,\
                "ceilingtex":ceilingtexlabels}
    return contexts

def return_colormapping():
    """
     Each object is rendered in a specific color by the object occupancy tracking shader with z
     encoded with exactly these colors, we use these mappings for decoding them in the code later
     """
    z=0.7
    unique_colors = np.array([[1,1,1], [z,z,z], [1,1,z], [1,z,1], [z,1,1], [1,z,0], [0,1,z],
            [z,0,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,z,z], [z,1,z]])
    unique_colors *= 255
    unique_colors = unique_colors.astype(int)
    return unique_colors
