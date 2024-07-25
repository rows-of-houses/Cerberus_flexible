import numpy as np
import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cpin
import os

### SETTINGS ###
urdf_filename = "/home/dmitry/Documents/unitree_ros/robots/a1_description/urdf/a1.urdf"
leg_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

### UTILS ###
def euler_to_rot(euler):
    r = euler[0]
    p = euler[1]
    y = euler[2]

    Ry = cs.SX.zeros((3, 3))
    Ry[0, 0] = cs.SX.cos(y)
    Ry[0, 1] = -cs.SX.sin(y)
    Ry[0, 2] = 0.0
    Ry[1, 0] = cs.SX.sin(y)
    Ry[1, 1] = cs.SX.cos(y)
    Ry[1, 2] = 0.0
    Ry[2, 0] = 0.0
    Ry[2, 1] = 0.0
    Ry[2, 2] = 1.0

    Rp = cs.SX.zeros((3, 3))
    Rp[0, 0] = cs.SX.cos(p)
    Rp[0, 1] = 0.0
    Rp[0, 2] = cs.SX.sin(p)
    Rp[1, 0] = 0.0
    Rp[1, 1] = 1.0
    Rp[1, 2] = 0.0
    Rp[2, 0] = -cs.SX.sin(p)
    Rp[2, 1] = 0.0
    Rp[2, 2] = cs.SX.cos(p)

    Rr = cs.SX.zeros((3, 3))
    Rr[0, 0] = 1.0
    Rr[0, 1] = 0.0
    Rr[0, 2] = 0.0
    Rr[1, 0] = 0.0
    Rr[1, 1] = cs.SX.cos(r)
    Rr[1, 2] = -cs.SX.sin(r)
    Rr[2, 0] = 0.0
    Rr[2, 1] = cs.SX.sin(r)
    Rr[2, 2] = cs.SX.cos(r)

    R = Ry @ Rp @ Rr

    return R

def mtx_w_to_euler_dot(euler):
    r = euler[0]
    p = euler[1]
    y = euler[2]

    mtx = cs.SX.zeros((3, 3))

    mtx[0, 0] = 1.0
    mtx[0, 1] = (cs.SX.sin(p) * cs.SX.sin(r)) / cs.SX.cos(p)
    mtx[0, 2] = (cs.SX.cos(r) * cs.SX.sin(p)) / cs.SX.cos(p)
    mtx[1, 0] = 0.0
    mtx[1, 1] = cs.SX.cos(r)
    mtx[1, 2] = -cs.SX.sin(r)
    mtx[2, 0] = 0.0
    mtx[2, 1] = cs.SX.sin(r) / cs.SX.cos(p)
    mtx[2, 2] = cs.SX.cos(r) / cs.SX.cos(p)

    return mtx

def skew_symmetric(w, lib=cs.SX):
    mtx = lib.zeros((3, 3))
    mtx[0, 0] = lib.zeros(1)[0]
    mtx[0, 1] = -w[2]
    mtx[0, 2] =  w[1]
    mtx[1, 0] =  w[2]
    mtx[1, 1] = lib.zeros(1)[0]
    mtx[1, 2] = -w[0]
    mtx[2, 0] = -w[1]
    mtx[2, 1] =  w[0]
    mtx[2, 2] = lib.zeros(1)[0]
    return mtx
### UTILS ###

def generate_kinematics(urdf_filename, leg_frames):
    def get_parent_joints(cmodel, frame_index):
        joints = [cmodel.frames[frame_index].parentJoint]
        while frame_index != 0:
            frame_index = cmodel.frames[frame_index].parentFrame
            joint = cmodel.frames[frame_index].parentJoint
            if joint != joints[-1]:
                joints.append(joint)

        joints = [a-1 for a in joints[::-1][1:]]

        return joints

    def get_frame_names(cmodel):
        return [(i, frame.name) for i, frame in enumerate(cmodel.frames.tolist())]
    
    model = pin.buildModelFromUrdf(urdf_filename)

    # adjusting parameters to make functions give the same values as initial ones.
    # IDK why but ShuoYang used wrond kinematic parameters.
    for i, leg_frame in enumerate(leg_frames):
        frame_id = model.getFrameId(leg_frame)
        joint_id = model.frames[frame_id].parentJoint    
        model.jointPlacements[joint_id].translation[2] = -0.21 # initially in URDF it is -0.20

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    rhos = cs.SX.sym("rhos", len(leg_frames)) # optimizable kinematic parameters
    cq = cs.SX.sym("joints", cmodel.nq) #joint angles

    # setting symbolic displacements for leg frames
    for i, leg_frame in enumerate(leg_frames):
        frame_id = model.getFrameId(leg_frame)
        trans = cs.SX.zeros(3)
        trans[2] = -rhos[i]
        
        cmodel.frames[frame_id].placement.translation = trans    
        
    cpin.forwardKinematics(cmodel, cdata, cq)

    fk_list = []
    jac_list = []
    dfk_drho_list = []
    dJ_dq_list = []
    dJ_drho_list = []

    for i, leg_frame in enumerate(leg_frames):
            
        frame_id = model.getFrameId(leg_frame)
        parent_joints = get_parent_joints(cmodel, frame_id)

        fk = cpin.updateFramePlacement(cmodel, cdata, frame_id).translation
        jac = cs.jacobian(cpin.updateFramePlacement(cmodel, cdata, frame_id).translation, cq[parent_joints])
        dfk_drho = cs.jacobian(cpin.updateFramePlacement(cmodel, cdata, frame_id).translation, rhos[i])
        dJ_dq = cs.jacobian(jac, cq[parent_joints])
        dJ_drho = cs.jacobian(jac, rhos[i])
        
        fk_list.append(fk)
        jac_list.append(jac)
        dfk_drho_list.append(dfk_drho)
        dJ_dq_list.append(dJ_dq)
        dJ_drho_list.append(dJ_drho)
        
    fk_function = cs.Function("fk", \
                            [cq, rhos], \
                            fk_list)

    jac_function = cs.Function("J", \
                            [cq, rhos], \
                            jac_list)

    dfk_drho_function = cs.Function("dfk_drho", \
                            [cq, rhos], \
                            dfk_drho_list)

    dJ_dq_function = cs.Function("dJ_dq", \
                            [cq, rhos], \
                            dJ_dq_list)

    dJ_drho_function = cs.Function("dJ_drho", \
                            [cq, rhos], \
                            dJ_drho_list)

    os.makedirs("src", exist_ok=True)
    os.makedirs("shared", exist_ok=True)

    os.chdir("src")

    fk_function.generate()
    jac_function.generate()
    dfk_drho_function.generate()
    dJ_dq_function.generate()
    dJ_drho_function.generate()
    
    return fk_function, jac_function # needed for EKF

def generate_ekf(fk_function, jac_function):
    
    ### GENERATE PROCESS FUNCTIONS ###
    
    def sipo_process_dyn(x, u):
        pos = x[0:3]
        vel = x[3:6]
        euler = x[6:9]

        # ba = x[21:24]
        # bg = x[24:27]

        w = u[0:3] # - bg
        a = u[3:6] # - ba

        deuler = mtx_w_to_euler_dot(euler) @ w

        R = euler_to_rot(euler)

        gravity = cs.SX(np.array([0, 0, 9.8]))
        acc = R @ a - gravity
        

        return cs.vertcat(vel, acc, deuler, cs.SX.zeros(12), cs.SX(1.0))

    def proc_func(xn, un, un1, dt):
        k1 = sipo_process_dyn(xn, un)
        k2 = sipo_process_dyn(xn + dt * k1 / 2, (un + un1) / 2)
        k3 = sipo_process_dyn(xn + dt * k2 / 2, (un + un1) / 2)
        k4 = sipo_process_dyn(xn + dt * k3, un1)

        xn1 = xn + (1 / 6.0) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

        return xn1
    
    x_codegen_arg_casadi = cs.SX.sym("x_codegen_arg_casadi", 22)
    u0_arg_casadi = cs.SX.sym("u0_arg_casadi", 7)
    u1_arg_casadi = cs.SX.sym("u1_arg_casadi", 7)
    dt_arg_casadi = cs.SX.sym("dt_arg_casadi", 1)
    
    proc_func_value = proc_func(x_codegen_arg_casadi, u0_arg_casadi, u1_arg_casadi, dt_arg_casadi)

    proc_function = cs.Function("process", \
                                [x_codegen_arg_casadi, u0_arg_casadi, u1_arg_casadi, dt_arg_casadi], 
                                [proc_func_value])
    
    proc_jac_function = cs.Function("process_jac", \
                            [x_codegen_arg_casadi, u0_arg_casadi, u1_arg_casadi, dt_arg_casadi], \
                            [cs.jacobian(proc_func_value, x_codegen_arg_casadi)])
    
    proc_function.generate()
    proc_jac_function.generate()


    ### GENERATE MEASUREMENT FUNCTIONS ###
            
    def measurement_func(x, wk, phik, dphik):
        param = cs.SX.ones(4) * 0.21
        pos = x[0:3]
        vel = x[3:6]
        euler = x[6:9]
        R_er = euler_to_rot(euler)

        foot_pos = x[9:21]

        # bg = x[24:27]
        bg = cs.SX.zeros(3)

        meas_residual = cs.SX.zeros(24)

        p_rf = fk_function(phik, param)
        J_rf = jac_function(phik, param)

        for i in range(len(leg_frames)):
            av = dphik[i * 3: i * 3 + 3]
            p_rf_one = p_rf[i]
            J_rf_one = J_rf[i]
            w_k_no_bias = wk - bg
            leg_v = (J_rf_one @ av).reshape((3, 1)) + skew_symmetric(w_k_no_bias) @ p_rf_one
            meas_residual[i * 6 : i * 6 + 3] = (p_rf_one - R_er.T @ (foot_pos[i * 3 : i * 3 + 3] - pos))
            meas_residual[i * 6 + 3 : i * 6 + 6] = (vel + (R_er @ leg_v))
        
        return meas_residual

    x = cs.SX.sym("x", 22)
    wk = cs.SX.sym("wk", 3)
    phik = cs.SX.sym("phik", 12)
    dphik = cs.SX.sym("dphik", 12)
    
    measurement_func_result = measurement_func(x, wk, phik, dphik)
    
    meas_function = cs.Function("meas", \
                                [x, wk, phik, dphik], \
                                [measurement_func_result])
    
    meas_jac_function = cs.Function("meas_jac", \
                                [x, wk, phik, dphik], \
                                [cs.jacobian(measurement_func_result, x)])

    meas_function.generate()
    meas_jac_function.generate()

if __name__ == "__main__":
    fk_function, jac_function = generate_kinematics(urdf_filename=urdf_filename, 
                                                    leg_frames=leg_frames)
    generate_ekf(fk_function, jac_function)
    
    os.system('gcc -fPIC -O3 -shared fk.c -o ../shared/fk.so')
    os.system('gcc -fPIC -O3 -shared J.c -o ../shared/J.so')
    os.system('gcc -fPIC -O3 -shared dfk_drho.c -o ../shared/dfk_drho.so')
    os.system('gcc -fPIC -O3 -shared dJ_dq.c -o ../shared/dJ_dq.so')
    os.system('gcc -fPIC -O3 -shared dJ_drho.c -o ../shared/dJ_drho.so')
    
    os.system('gcc -fPIC -O3 -shared process.c -o ../shared/process.so')
    os.system('gcc -fPIC -O3 -shared process_jac.c -o ../shared/process_jac.so')
    os.system('gcc -fPIC -O3 -shared meas.c -o ../shared/meas.so')
    os.system('gcc -fPIC -O3 -shared meas_jac.c -o ../shared/meas_jac.so')