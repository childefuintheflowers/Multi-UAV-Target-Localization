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

# import RflySim APIs
# import self as self
import torch
import PX4MavCtrlV4 as PX4MavCtrl
import VisionCaptureApi
import multiprocessing as mp
import threading

vis = VisionCaptureApi.VisionCaptureApi()



# VisionCaptureApi 中的配置函数
vis.jsonLoad(jsonPath='/Config3.json')  # 加载Config.json中的传感器配置文件

isSuss = vis.sendReqToUE4()  # 向RflySim3D发送取图请求，并验证
if not isSuss:  # 如果请求取图失败，则退出
    sys.exit(0)
vis.startImgCap(True)  # 开启共享内存取图


vis.sendUE4PosScale(100, 30, 0, [3, 0, 0], [0, 0, np.pi/2], [1, 1, 1])

# Send command to UE4 Window 1 to change resolution
vis.sendUE4Cmd(b'r.setres 720x405w', 0)  # 设置UE4窗口分辨率，注意本窗口仅限于显示，取图分辨率在json中配置，本窗口设置越小，资源需求越少。
vis.sendUE4Cmd(b't.MaxFPS 30', 0)  # 设置UE4最大刷新频率，同时也是取图频率
time.sleep(2)


# Create MAVLink control API instance
mav1 = PX4MavCtrl.PX4MavCtrler(20100)
mav2 = PX4MavCtrl.PX4MavCtrler(20102)
mav3 = PX4MavCtrl.PX4MavCtrler(20104)
# Init MAVLink data receiving loop
mav1.InitMavLoop()
mav2.InitMavLoop()
mav3.InitMavLoop()

print("Simulation Start.")
print("Enter Offboard mode.")
time.sleep(2)

mav1.initOffboard()
mav2.initOffboard()
mav3.initOffboard()


mav1.SendMavArm(True)  # Arm the drone
mav2.SendMavArm(True)
mav3.SendMavArm(True)


lastTime = time.time()
startTime = time.time()
# time interval of the timer
timeInterval = 1 / 30.0  # here is 0.0333s (30Hz)
flag = 0

# parameters
width = 640
height = 480
channel = 4
min_prop = 0.000001
max_prop = 0.3
K_z = 0.003 * 640 / height
K_yawrate = 0.005 * 480 / width
num = 0
lastClock = time.time()





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

def global_situation(uav_pos, uav_ang, data):
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
                           [0, 0, data[4] * data[4]]])/36
    cov_rot = rotate_matrix(0,0,-data[6])
    global_cov = np.dot(np.dot(cov_rot, cov_matrix), np.transpose(cov_rot))


    return situatition, global_cov



# build the model from a config file and a checkpoint file
model1 = init_model(args.config, args.checkpoint, device=args.device)
model2 = init_model(args.config, args.checkpoint, device=args.device)
model3 = init_model(args.config, args.checkpoint, device=args.device)


def UAV1():
    num1 = 0
    flag1 = 0
    lastTime1 = time.time()
    lastClock1 = time.time()
    while True:
        lastTime1 = lastTime1 + timeInterval
        sleepTime1 = lastTime1 - time.time()
        if sleepTime1 > 0:
            time.sleep(sleepTime1)  # sleep until the desired clock
        else:
            lastTime1 = time.time()
        # The following code will be executed 30Hz (0.0333s)

        num1 = num1 + 1
        if num1 % 100 == 0:
            tiem1 = time.time()
            print('UAV 1 MainThreadFPS: ' + str(100 / (tiem1 - lastClock1)))
            lastClock1 = tiem1

            # TargetPos = [40, 5, -1]
            # vis.sendUE4PosScale(100, 2050, 0, TargetPos, [0, 0, np.pi/2], [1, 1, 1])
            time.sleep(0.5)

        if time.time() - startTime > 5 and flag1 == 0:
            # The following code will be executed at 5s
            print("5s, Arm the drone")
            flag1 = 1
            mav1.SendPosNED(-3, 0, -3, 0)  # Fly to target position [0, 0, -5], i.e., take off to 5m




        # if time.time() - startTime > 15 and flag1 == 1:
        #     flag1 = 1
        #     mav1.SendPosNED(-3, 0, -3, 0)  # Fly to target position [0, 0, -5], i.e., take off to 5m


        if vis.hasData[0]:
            img_bgr1 = vis.Img[0]
            time.sleep(0.05)
            result, data1 = uav_multi_mono_3d_detector(model1, img_bgr1, args.ann)
            if result[0]['result']['scores_3d'].data.numel() > 0:
                aa = result[0]['scores_3d'].data
                _, a = torch.max(aa, 0)
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
                sit, cov = global_situation(mav1.uavPosNED, mav1.uavAngEular, first_data)
            else:
                sit, cov =[]
            show_wind_mono3d(
                data1,
                result,
                img_bgr1,
                args.out_dir,
                args.score_thr,
                windowname='img1',
                show=True,
                snapshot=args.snapshot,
            )



def UAV2():
    num2 = 0
    flag2 = 0
    lastTime2 = time.time()
    lastClock2 = time.time()
    while True:
        lastTime2 = lastTime2 + timeInterval
        sleepTime2 = lastTime2 - time.time()
        if sleepTime2 > 0:
            time.sleep(sleepTime2)  # sleep until the desired clock
        else:
            lastTime2 = time.time()
        # The following code will be executed 30Hz (0.0333s)

        num2 = num2 + 1
        if num2 % 100 == 0:
            tiem2 = time.time()
            print('UAV 2 MainThreadFPS: ' + str(100 / (tiem2- lastClock2)))
            lastClock2 = tiem2

            # TargetPos = [40, 5, -1]
            # vis.sendUE4PosScale(100, 2050, 0, TargetPos, [0, 0, np.pi/2], [1, 1, 1])
            time.sleep(0.5)

        if time.time() - startTime > 5 and flag2 == 0:
            # The following code will be executed at 5s
            print("5s, Arm the drone")
            flag2 = 1
            mav2.SendPosNED(-3, -0, -3, 0)  # Fly to target position [0, 0, -5], i.e., take off to 5m




        # if time.time() - startTime > 15 and flag2 == 1:
        #     flag2 = 1
        #     mav2.SendPosNED(-3, 0, -3, 0)  # Fly to target position [0, 0, -5], i.e., take off to 5m


        if vis.hasData[1]:
            img_bgr2 = vis.Img[1]
            time.sleep(0.05)
            result, data2 = uav_multi_mono_3d_detector(model2, img_bgr2, args.ann)
            if result[0]['result']['scores_3d'].data.numel() > 0:
                aa = result[0]['scores_3d'].data
                _, a = torch.max(aa, 0)
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
                sit, cov = global_situation(mav1.uavPosNED, mav1.uavAngEular, first_data)
            else:
                sit, cov =[]
            show_wind_mono3d(
                data2,
                result,
                img_bgr2,
                args.out_dir,
                args.score_thr,
                windowname='img2',
                show=True,
                snapshot=args.snapshot,
            )



def UAV3():
    num3 = 0
    flag3 = 0
    lastTime3 = time.time()
    lastClock3 = time.time()
    while True:
        lastTime3 = lastTime3 + timeInterval
        sleepTime3 = lastTime3 - time.time()
        if sleepTime3 > 0:
            time.sleep(sleepTime3)  # sleep until the desired clock
        else:
            lastTime3 = time.time()
        # The following code will be executed 30Hz (0.0333s)

        num3 = num3 + 1
        if num % 100 == 0:
            tiem3 = time.time()
            print('UAV 3 MainThreadFPS: ' + str(100 / (tiem3 - lastClock3)))
            lastClock3 = tiem3

            # TargetPos = [40, 5, -1]
            # vis.sendUE4PosScale(100, 2050, 0, TargetPos, [0, 0, np.pi/2], [1, 1, 1])
            time.sleep(0.5)

        if time.time() - startTime > 5 and flag3 == 0:
            # The following code will be executed at 5s
            print("5s, Arm the drone")
            flag3 = 1
            mav3.SendPosNED(-6, -3, -3, 0)  # Fly to target position [0, 0, -5], i.e., take off to 5m




        # if time.time() - startTime > 15 and flag3 == 1:
        #     flag3 = 1
        #     mav3.SendPosNED(-3, 0, -3, 0)  # Fly to target position [0, 0, -5], i.e., take off to 5m


        if vis.hasData[2]:
            img_bgr3 = vis.Img[2]
            time.sleep(0.05)
            result, data3 = uav_multi_mono_3d_detector(model3, img_bgr3, args.ann)
            if result[0]['result']['scores_3d'].data.numel() > 0:
                aa = result[0]['scores_3d'].data
                _, a = torch.max(aa, 0)
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
                sit, cov = global_situation(mav1.uavPosNED, mav1.uavAngEular, first_data)
            else:
                sit, cov =[]
            show_wind_mono3d(
                data3,
                result,
                img_bgr3,
                args.out_dir,
                args.score_thr,
                windowname='img3',
                show=True,
                snapshot=args.snapshot,
            )


if __name__ == '__main__':
    uav1 = threading.Thread(target=UAV1)
    uav2 = threading.Thread(target=UAV2)
    uav3 = threading.Thread(target=UAV3)

    uav1.start()
    uav2.start()
    uav3.start()
