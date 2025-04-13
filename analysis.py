# -*- coding: UTF-8 -*-
from key_controller import Controller, calculate_angle, get_pos

# Metablic equivalent to calculate calories: kcal = MET * BODY_WEIGHT(kg) * TIME(hr)
METs = {'stand': 1.3, 'pushup': 3.8, 'situp': 2.8, 'squat': 5.0}
BODY_WEIGHT = 65

def to_xy(data):
    return data[0], data[1]

def calculate_angle_index(data, a, b, c):
    return calculate_angle(to_xy(data[a]), to_xy(data[b]), to_xy(data[c]))

def get_joint_angles(data):

    knee_angle = 180 - calculate_angle(get_pos(data, 'foot'),
                                 get_pos(data, 'knee'), get_pos(data, 'hip'))
    hip_angle = 180 - calculate_angle(get_pos(data, 'knee'),
                                 get_pos(data, 'hip'), get_pos(data, 'shoulder'))

    left_arm_angle = 180 - calculate_angle_index(data, 11, 13, 15)
    right_arm_angle = 180 - calculate_angle_index(data, 12, 14, 16)

    return left_arm_angle, right_arm_angle, knee_angle, hip_angle


def calculate_energy(controller: Controller, time):
    if controller.pushup_state is not None:
        MET = METs['pushup']
    elif controller.situp_state is not None:
        MET = METs['situp']
    elif controller.squat_state is not None:
        MET = METs['squat']
    else:
        MET = METs['stand']
    calorie = MET * BODY_WEIGHT * time / 3600 # kcal
    return calorie