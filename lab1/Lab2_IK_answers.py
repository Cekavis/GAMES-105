import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    max_iter = 100
    eps = 0.01

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    length = [np.linalg.norm(joint_positions[i] - joint_positions[j]) for i, j in zip(path[1:], path[:-1])]
    pos = joint_positions.copy()
    rot = joint_orientations.copy()

    # FABRIK
    for i in range(max_iter):
        pos[path[-1]] = target_pose
        for j in range(len(path) - 2, -1, -1):
            x, y = path[j], path[j + 1]
            pos[x] = pos[y] + (pos[x] - pos[y]) / np.linalg.norm(pos[x] - pos[y]) * length[j]
        pos[path[0]] = joint_positions[path[0]]
        for j in range(len(path) - 1):
            x, y = path[j], path[j + 1]
            pos[y] = pos[x] + (pos[y] - pos[x]) / np.linalg.norm(pos[y] - pos[x]) * length[j]
        if np.linalg.norm(pos[path[-1]] - target_pose) < eps:
            break

    print('FABRIK iter:', i + 1)

    def get_rot_from_vecs(vec0, vec1):
        """
        Get the rotation that rotates vec1 to vec0.
        """
        from numpy.linalg import norm
        vec0 = vec0 / norm(vec0)
        vec1 = vec1 / norm(vec1)
        cross = np.cross(vec0, vec1)
        dot = np.clip(np.dot(vec0, vec1), -1, 1)
        theta = np.arccos(dot)
        if np.allclose(cross, 0):
            cross = np.cross(vec0, [1, 0, 0])
        if np.allclose(cross, 0):
            cross = np.cross(vec0, [0, 1, 0])
        rot_vec = cross / norm(cross) * theta
        rot = R.from_rotvec(rot_vec)
        assert np.allclose(rot.apply(vec0), vec1)
        return rot

    
    # Update joint_orientations
    for p in [path1, path2]:
        for i in range(len(p) - 1):
            new_offset = pos[p[i]] - pos[p[i + 1]]
            old_offset = joint_positions[p[i]] - joint_positions[p[i + 1]]
            old_offset = R.from_quat(joint_orientations[p[i + 1]]).inv().apply(old_offset)
            joint_orientations[p[i + 1]] = get_rot_from_vecs(old_offset, new_offset).as_quat()

    vis = [i in path for i in range(len(joint_positions))]

    def update_pos(i):
        p = meta_data.joint_parent[i]
        if p == -1 or vis[i]:
            return
        vis[i] = True
        update_pos(p)

        offset = joint_positions[i] - joint_positions[p]
        offset = R.from_quat(rot[p]).inv().apply(offset)
        joint_orientations[i] = joint_orientations[p]
        pos[i] = pos[p] + R.from_quat(joint_orientations[p]).apply(offset)

    for i in range(len(joint_positions)):
        update_pos(i)
            
    joint_positions = pos
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    max_iter = 100
    eps = 0.01

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    path.append(meta_data.joint_name.index('lWrist_end'))
    length = [np.linalg.norm(joint_positions[i] - joint_positions[j]) for i, j in zip(path[1:], path[:-1])]
    pos = joint_positions.copy()
    rot = joint_orientations.copy()

    # FABRIK
    for i in range(max_iter):
        target_pose = joint_positions[meta_data.joint_name.index('RootJoint')] + (relative_x, 0, relative_z)
        target_pose[1] = target_height
        pos[path[-1]] = target_pose
        for j in range(len(path) - 2, -1, -1):
            x, y = path[j], path[j + 1]
            pos[x] = pos[y] + (pos[x] - pos[y]) / np.linalg.norm(pos[x] - pos[y]) * length[j]
        pos[path[0]] = joint_positions[path[0]]
        for j in range(len(path) - 1):
            x, y = path[j], path[j + 1]
            pos[y] = pos[x] + (pos[y] - pos[x]) / np.linalg.norm(pos[y] - pos[x]) * length[j]
        if np.linalg.norm(pos[path[-1]] - target_pose) < eps:
            break

    print('FABRIK iter:', i + 1)

    def get_rot_from_vecs(vec0, vec1):
        """
        Get the rotation that rotates vec1 to vec0.
        """
        from numpy.linalg import norm
        vec0 = vec0 / norm(vec0)
        vec1 = vec1 / norm(vec1)
        cross = np.cross(vec0, vec1)
        dot = np.clip(np.dot(vec0, vec1), -1, 1)
        theta = np.arccos(dot)
        if np.allclose(cross, 0):
            cross = np.cross(vec0, [1, 0, 0])
        if np.allclose(cross, 0):
            cross = np.cross(vec0, [0, 1, 0])
        rot_vec = cross / norm(cross) * theta
        rot = R.from_rotvec(rot_vec)
        assert np.allclose(rot.apply(vec0), vec1)
        return rot

    
    # Update joint_orientations
    path = list(reversed(path))
    for i in range(len(path) - 1):
        new_offset = pos[path[i]] - pos[path[i + 1]]
        old_offset = joint_positions[path[i]] - joint_positions[path[i + 1]]
        old_offset = R.from_quat(rot[path[i + 1]]).inv().apply(old_offset)
        joint_orientations[path[i + 1]] = get_rot_from_vecs(old_offset, new_offset).as_quat()
    
    return pos, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations