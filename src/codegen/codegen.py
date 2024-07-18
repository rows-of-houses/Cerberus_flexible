import numpy as np
import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cpin
import os


### DEBUG FUNCTIONS ###
def get_parent_joints(cmodel, frame_index):
    '''
    Get all the joint indeces (-1) that affect specified frame
    '''
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


### SETTINGS ###
urdf_filename = "/home/dmitry/Documents/unitree_ros/robots/a1_description/urdf/a1.urdf"
leg_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


model = pin.buildModelFromUrdf(urdf_filename)
cmodel = cpin.Model(model)
cdata = cmodel.createData()

rhos = cs.SX.sym("rhos", len(leg_frames)) # optimizable kinematic parameters
cq = cs.SX.sym("joints", cmodel.nq) #joint angles

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
    jac = cs.jacobian(cpin.updateFramePlacement(cmodel, cdata, frame_id).translation, cq[parent_joints]).reshape((1, -1))
    dfk_drho = cs.jacobian(cpin.updateFramePlacement(cmodel, cdata, frame_id).translation, rhos[i])
    dJ_dq = cs.jacobian(jac, cq[parent_joints]).reshape((1, -1))
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

os.mkdir("src")
os.mkdir("shared")

os.chdir("src")

fk_function.generate()
jac_function.generate()
dfk_drho_function.generate()
dJ_dq_function.generate()
dJ_drho_function.generate()

os.system('gcc -fPIC -shared fk.c -o ../shared/fk.so')
os.system('gcc -fPIC -shared J.c -o ../shared/J.so')
os.system('gcc -fPIC -shared dfk_drho.c -o ../shared/dfk_drho.so')
os.system('gcc -fPIC -shared dJ_dq.c -o ../shared/dJ_dq.so')
os.system('gcc -fPIC -shared dJ_drho.c -o ../shared/dJ_drho.so')
