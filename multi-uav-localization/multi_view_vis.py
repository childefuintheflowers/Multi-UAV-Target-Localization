import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab, uav_multi_mono_3d_detector,
                          show_wind_mono3d)
import time
import cv2.cv2 as cv2
import sys
import cv2 as cv
import math
import torch
import PX4MavCtrlV4 as PX4MavCtrl
import VisionCaptureApi
import pandas as pd
import csv

vis = VisionCaptureApi.VisionCaptureApi()
vis.sendUE4Cmd(b'RflyChangeMapbyName MatchScene_606')

uav1_pos = [-0, 0-46, -5-1]
uav1_ang = [0,0,0]
vis.sendUE4Pos(1,3,0,uav1_pos,uav1_ang)

uav2_pos = [0, -3-46, -5-1]
uav2_ang = [0,0,np.pi/6]
vis.sendUE4Pos(2,3,0,uav2_pos,uav2_ang)

uav3_pos = [0, 3-46, -5-1]
uav3_ang = [0,0,-np.pi/6]
vis.sendUE4Pos(3,3,0,uav3_pos,uav3_ang)

# VisionCaptureApi 中的配置函数
vis.jsonLoad(jsonPath='/Config3.json')  # 加载Config.json中的传感器配置文件

isSuss = vis.sendReqToUE4()  # 向RflySim3D发送取图请求，并验证
if not isSuss:  # 如果请求取图失败，则退出
    sys.exit(0)
vis.startImgCap(True)  # 开启共享内存取图

target_pos = [7, 0-46, -1.054]
target_ang = [0, 0, 0]
vis.sendUE4PosScale(100, 30, 0, target_pos, target_ang, [1, 1, 1])

# Send command to UE4 Window 1 to change resolution
vis.sendUE4Cmd(b'r.setres 720x405w', 0)  # 设置UE4窗口分辨率，注意本窗口仅限于显示，取图分辨率在json中配置，本窗口设置越小，资源需求越少。
vis.sendUE4Cmd(b't.MaxFPS 30', 0)  # 设置UE4最大刷新频率，同时也是取图频率
time.sleep(2)

lastTime = time.time()
startTime = time.time()
# time interval of the timer
timeInterval = 1 / 30.0  # here is 0.0333s (30Hz)
lastClock = time.time()
t1 = 0

# Initialize parameters
initial_state = np.array([0, 0, 0, 1, 1, 1])
initial_covariance = np.eye(6)
transition_matrix = np.eye(6)
transition_matrix[:3, 3:] = np.eye(3)
observation_matrix = np.array([
    [1, 0, 0, 0, 0, 0],  # x
    [0, 1, 0, 0, 0, 0],  # y
    [0, 0, 1, 0, 0, 0],  # z
])
process_noise_covariance = np.eye(6) * 0.001
observation_noise_covariance = np.eye(6) * 0.1



# 设置相应参数和加载模型

parser = ArgumentParser()
parser.add_argument('--image', type=str, default=r'D:\open-mmlab\mmdetection3d-master\demo\data\nuscenes\21.jpg', help='image file')
parser.add_argument('--ann', type=str, default=r'D:\open-mmlab\mmdetection3d-master\demo\data\nuscenes\21.coco.json', help='ann file')
parser.add_argument('--config', type=str, default=r'D:\open-mmlab\mmdetection3d-master\configs\pgd\pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py', help='Config file')
parser.add_argument('--checkpoint', type=str, default=r'D:\open-mmlab\mmdetection3d-master\checkpoints\pgd\pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d_20211116_195350-f4b5eec2.pth', help='Checkpoint file')
parser.add_argument(
    '--device', default='cuda:0', help='Device used for inference')
parser.add_argument(
    '--score-thr', type=float, default=0.07, help='bbox score threshold')
parser.add_argument(
    '--out-dir', type=str, default='demo', help='dir to save results')
parser.add_argument(
    '--show',
    action='store_true',
    help='show online visualization results')
parser.add_argument(
    '--snapshot',
    action='store_true',
    help='whether to save online visualization results')
args = parser.parse_args()


def rotate_matrix(roll, pitch, yaw):
    matrix_roll = [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]]
    matrix_pitch = [[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]]
    matrix_yaw = [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    rotation_matrix = np.dot(matrix_roll, np.dot(matrix_pitch, matrix_yaw))
    return rotation_matrix

def global_situation(uav_pos, uav_ang, data, score):
    '''
    Args:
        uav_pos:xyz对应北东地
        uav_ang:roll-pitch-yaw
        data:3D检测结果中的[x, y, z, w, h, l, theta]即为（中心坐标， 长宽高， 方向角），其中xyz为东地北，theta东方向为0°方向，逆时针为正方向
    '''
    rot = rotate_matrix(uav_ang[0], uav_ang[1], uav_ang[2])
    detect_pos = np.array([data[2], data[0], data[1]]) + np.array([0.3, 0, data[4]/2]) # 将检测结果的东地北转换为rflysim的北东地,再加上补偿
    # 世界坐标系下的坐标
    situatition = uav_pos + np.dot(detect_pos, rot)
    # 协方差矩阵
    cov_matrix = np.array([[data[5] * data[5], 0, 0],
                           [0, data[3] * data[3], 0],
                           [0, 0, data[4] * data[4]]])/(36*score*score)
    cov_rot = rotate_matrix(0,0,-data[6]-uav_ang[2])
    global_cov = np.dot(np.dot(cov_rot, cov_matrix), np.transpose(cov_rot))


    return situatition, global_cov

def uav_result(img_bgr, model, uavpos, uavang, name = 'window'):
    result, data = uav_multi_mono_3d_detector(model, img_bgr, args.ann)

    show_wind_mono3d(
        data,
        result,
        img_bgr,
        args.out_dir,
        args.score_thr,
        windowname=name,
        show=True,
        snapshot=args.snapshot,
    )

    if result[0]['scores_3d'].data.numel() > 0:
        # 取置信度最大的检测结果的中心坐标值
        aa = result[0]['scores_3d'].data
        a_m, a = torch.max(aa, 0)
        a = a.item()

        # 所有框的中心坐标、长宽高、鸟瞰视角数据
        b = result[0]['boxes_3d'].gravity_center
        c = result[0]['boxes_3d'].dims
        d = result[0]['boxes_3d'].bev
        # 置信度最大的一个框的中心坐标、长宽高、鸟瞰视角数据
        first_center = b[a].numpy()
        first_dims = c[a].numpy()
        first_bev = d[a].numpy()
        # 整合下来后续计算有用的数据[x, y, z, l, w, h, theta]即为（中心坐标， 长宽高， 方向角），其中xyz为东地北
        first_data = np.append(first_center, first_dims)
        first_data = np.append(first_data, first_bev[4])
        sit, cov = global_situation(uavpos, uavang, first_data, a_m)
        # print(name, '无人机的预测坐标为:', sit)
        # print(name, '无人机的预测协方差为:', cov)
        return sit, cov
    else:
        sit, cov = []
        # print(name, '无人机未能检测到目标')
        return sit, cov

def get_distance(a,b):
    c = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]])
    return np.linalg.norm(c)

def result_fusion():

    pass


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, transition_matrix_function, observation_matrix, process_noise_covariance_function, observation_noise_covariance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.transition_matrix_function = transition_matrix_function
        self.observation_matrix = observation_matrix
        self.process_noise_covariance_function = process_noise_covariance_function
        self.observation_noise_covariance = observation_noise_covariance

    def predict(self, dt):
        transition_matrix = self.transition_matrix_function(dt)
        process_noise_covariance = self.process_noise_covariance_function(dt)
        self.state = transition_matrix @ self.state
        self.covariance = transition_matrix @ self.covariance @ transition_matrix.T + process_noise_covariance
        return self.state[0:3], self.covariance[0:3, 0:3]

    def update(self, observation, observation_noise_covariance):
        innovation = observation - self.observation_matrix @ self.state
        innovation_covariance = self.observation_matrix @ self.covariance @ self.observation_matrix.T + observation_noise_covariance
        kalman_gain = self.covariance @ self.observation_matrix.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ self.observation_matrix @ self.covariance


def transition_matrix_function(dt):
    transition_matrix = np.eye(6)
    transition_matrix[:3, 3:] = np.eye(3) * dt
    return transition_matrix

def process_noise_covariance_function(dt):
    q = 0.001
    return np.diag([q*dt**3/3, q*dt**3/3, q*dt**3/3, q*dt, q*dt, q*dt])


# build the model from a config file and a checkpoint file
model1 = init_model(args.config, args.checkpoint, device=args.device)
model2 = init_model(args.config, args.checkpoint, device=args.device)
model3 = init_model(args.config, args.checkpoint, device=args.device)



if __name__ == '__main__':
    # Start a endless loop with 30Hz, timeInterval=1/30.0
    # sit1, sit2, sit3, cov1, cov2, cov3, sit, cov = np.array([])
    df = pd.DataFrame(columns=['targetpos','sit1', 'sit2', 'sit3', 'sit', 'sit_kf', 'err1', 'err2', 'err3', 'err', 'err_kf', 'cov1', 'cov2', 'cov3', 'cov', 'cov_kf'])

    kf = KalmanFilter(initial_state, initial_covariance, transition_matrix_function, observation_matrix,
                      process_noise_covariance_function, observation_noise_covariance)
    while True:
        # 1号视角，初始坐标即为世界坐标的(0, 0, 0)
        if vis.hasData[0]:
            img_bgr1 = vis.Img[0]
            time.sleep(0.05)
            sit1, cov1 = uav_result(img_bgr1, model1, uav1_pos, uav1_ang, name='UAV1')

        # 2号视角，初始坐标为相对于1号视角的(0,2,0)
        if vis.hasData[1]:
            img_bgr2 = vis.Img[1]
            time.sleep(0.05)
            sit2, cov2 = uav_result(img_bgr2, model2, uav2_pos, uav2_ang, name='UAV2')

        # 3号视角，初始坐标为相对于1号视角的(2,0,0)
        if vis.hasData[2]:
            img_bgr3 = vis.Img[2]
            time.sleep(0.05)
            sit3, cov3 = uav_result(img_bgr3, model3, uav3_pos, uav3_ang, name='UAV3')

        if sit1.size * cov1.size * sit2.size * cov2.size * sit3.size * cov3.size > 0:
            if t1 == 0:
                dt = 1
                t1 = time.time()
            else:
                dt = 0.1
                # dt = time.time() - t1
                t1 = time.time()
            cov = np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2) + np.linalg.inv(cov3))
            sit = np.dot((np.dot(sit1, np.linalg.inv(cov1)) + np.dot(sit2, np.linalg.inv(cov2)) + np.dot(sit3, np.linalg.inv(cov3))), cov)
            sit_kf, cov_kf = kf.predict(dt)
            kf.update(sit, cov)
            err1 = get_distance(sit1, target_pos)
            err2 = get_distance(sit2, target_pos)
            err3 = get_distance(sit3, target_pos)
            err = get_distance(sit, target_pos)
            err_kf = get_distance(sit_kf, target_pos)
            print(sit1, '1号误差', err1)
            print(sit2, '2号误差', err2)
            print(sit3, '3号误差', err3)
            print(sit, '融合后误差', err)
            print(sit_kf, 'kf误差', err_kf)
            print('1号协方差:', cov1)
            print('2号协方差:', cov2)
            print('3号协方差:', cov3)
            print('融合后协方差:', cov)
            print('kf协方差:', cov_kf)
            new_data = [target_pos,sit1, sit2, sit3, sit, sit_kf, err1, err2, err3, err, err_kf, cov1, cov2, cov3, cov, cov_kf]
            df.loc[len(df)] = new_data
            df.to_csv('realtime_data2.csv', index=False)
            target_pos[0] = target_pos[0] + 0.1
            vis.sendUE4PosScale(100, 30, 0, target_pos, target_ang, [1, 1, 1])





