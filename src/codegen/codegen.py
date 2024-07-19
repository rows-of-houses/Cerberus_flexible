import numpy as np
import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cpin
import os


### SETTINGS ###
urdf_filename = "/home/dmitry/Documents/unitree_ros/robots/a1_description/urdf/a1.urdf"
leg_frames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


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

# setting symbolic displacements for frames
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

os.system('gcc -fPIC -O3 -shared fk.c -o ../shared/fk.so')
os.system('gcc -fPIC -O3 -shared J.c -o ../shared/J.so')
os.system('gcc -fPIC -O3 -shared dfk_drho.c -o ../shared/dfk_drho.so')
os.system('gcc -fPIC -O3 -shared dJ_dq.c -o ../shared/dJ_dq.so')
os.system('gcc -fPIC -O3 -shared dJ_drho.c -o ../shared/dJ_drho.so')