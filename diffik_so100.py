import mujoco
import mujoco.viewer
import numpy as np
import time

integration_dt: float = 0.5

damping: float = 1e-4

gravity_compensation: bool = True

dt: float = 0.002

max_angvel = 0.0


def main() -> None:
    xml_path = 'models/scene.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)

    print("Joint Ranges:", model.jnt_range)

    data = mujoco.MjData(model)

    model.opt.timestep = dt

    site_id = model.site("attachment_site").id

    body_names = ['Fixed_Jaw', 'Lower_Arm', 'Moving Jaw', 'Rotation_Pitch', 'Upper_Arm', 'Wrist_Pitch_Roll', 'world']
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

    joint_names = ['Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
    dof_ids = np.array([model.joint(name).id for name in joint_names])

    base_rotation_id = model.joint('Rotation').id
    
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    key_id = model.key("home").id

    mocap_id = model.body("target").mocapid[0]

    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        z = r * np.sin(2 * np.pi * f * t) + 0.2

        return np.array([x, y, z])

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        
        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        idx = 0
        while viewer.is_running():
            step_start = time.time()

            x, y, z = circle(data.time, 0.13, 0.1, -0.2, 0.1)
            data.mocap_pos[mocap_id, 0:3] = (x, y, z)

            print('x, y, z', x, y, z)
            
            theta_deg = np.arctan2(x, -(y + 0.0452))
            data.qpos[base_rotation_id] = theta_deg
            
            print(theta_deg)


            # Position error.
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            # Orientation error.
            # print('data.site(site_id).xmat', data.site(site_id).xmat)
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            # print('site_quant', site_quat)

            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 2.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # print(jac)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # print('Before', q[dof_ids])
            # print('*model.jnt_range.T', *model.jnt_range.T)
            
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # q[base_rotation_id] = theta_deg
            # data.ctrl[base_rotation_id] = theta_deg

            # print('After', q[dof_ids])

            # Step the simulation.
            mujoco.mj_step(model, data)

            print('--------------------------------')
            idx = idx + 1
            # if idx == 200: break

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
