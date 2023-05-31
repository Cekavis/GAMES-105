import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    with open(bvh_file_path, 'r') as f:
        words = ' '.join(f.readlines()).split()
        joint_name = []
        joint_parent = []
        joint_offset = []
        parent = -1
        for i in range(len(words)):
            if words[i] == 'ROOT':
                joint_name.append(words[i+1])
                joint_parent.append(parent)
                parent = len(joint_name) - 1
                joint_offset.append([float(x) for x in words[i+4:i+7]])
            elif words[i] == 'JOINT':
                joint_name.append(words[i+1])
                joint_parent.append(parent)
                parent = len(joint_name) - 1
                joint_offset.append([float(x) for x in words[i+4:i+7]])
            elif words[i] == 'End':
                joint_name.append(joint_name[parent] + '_end')
                joint_parent.append(parent)
                parent = len(joint_name) - 1
                joint_offset.append([float(x) for x in words[i+4:i+7]])
            elif words[i] == '}':
                parent = joint_parent[parent]
            elif words[i] == 'MOTION':
                break
        joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    x = 0
    data = motion_data[frame_id]
    for i in range(len(joint_name)):
        p = joint_parent[i]
        if p == -1:
            joint_positions.append(data[x:x+3])
            x += 3
            joint_orientations.append(R.from_euler('XYZ', data[x:x+3], degrees=True).as_quat())
            x += 3
        elif joint_name[i].endswith('_end'):
            joint_positions.append(joint_positions[p] + R.from_quat(joint_orientations[p]).apply(joint_offset[i]))
            joint_orientations.append(joint_orientations[p])
        else:
            joint_positions.append(joint_positions[p] + R.from_quat(joint_orientations[p]).apply(joint_offset[i]))
            joint_orientations.append((R.from_quat(joint_orientations[p]) * R.from_euler('XYZ', data[x:x+3], degrees=True)).as_quat())
            x += 3
    
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)

    rotation = [R.from_euler('XYZ', [0, 0, 0], degrees=True)] * len(T_joint_name)
    for i in range(len(A_joint_name)-1, -1, -1):
        p = A_joint_parent[i]
        Ti = T_joint_name.index(A_joint_name[i])
        
        if p != -1:
            rotation[p] = R.align_vectors([A_joint_offset[i]], [T_joint_offset[Ti]])[0]
            rotation[i] = rotation[p].inv() * rotation[i]

    T_idx = []
    x = 0
    for i in range(len(T_joint_name)):
        if not T_joint_name[i].endswith('_end'):
            x += 3
        T_idx.append(x)

    motion_data = load_motion_data(A_pose_bvh_path)
    for f in range(len(motion_data)):
        retarget = motion_data[f].copy()
        x = 3
        for i in range(len(A_joint_name)):
            p = A_joint_parent[i]
            idx = T_idx[T_joint_name.index(A_joint_name[i])]
            if not A_joint_name[i].endswith('_end'):
                retarget[idx:idx+3] = (R.from_euler('XYZ', motion_data[f, x:x+3], degrees=True) * rotation[i]).as_euler('XYZ', degrees=True)
                x += 3
        motion_data[f] = retarget

    return motion_data
