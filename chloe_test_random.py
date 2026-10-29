import time
import mujoco
import mujoco.viewer
import numpy as np

# table centre 
TABLE_CENTRE = np.array([0.04, -0.29])
# ball centre of height 
BALL_Z = 0.145  # table top 0.12 + radius 0.025

# function for randomising ball 
def randomise_ball(model, data, seed):
    rng = np.random.default_rng(seed) # random generator. Same seed --> same numbers
    x = TABLE_CENTRE[0] + rng.uniform(-0.10, 0.10) # random x within +- 10cm of table centre
    y = TABLE_CENTRE[1] + rng.uniform(-0.08, 0.08) # random y within +- 8cm of table centre
    adr = model.jnt_qposadr[model.joint("target_ball_freejoint").id] # index where the ball's values start in qpos 
    data.qpos[adr:adr+3] = [x, y, BALL_Z] # set the ball position (x, y, z) 
    data.qpos[adr+3:adr+7] = [1, 0, 0, 0] # set ball orientation - quaternion with no rotation
    mujoco.mj_forward(model, data) # update body positions/camera without advancing time
    return np.array([x, y, BALL_Z]) # return the true ball position 


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    for seed in range(5): # test seeds 0 through to 4 
        mujoco.mj_resetData(model, data) # reset everything to the starting state 
        a = randomise_ball(model, data, seed) # place the ball, save the position as "a"
        mujoco.mj_resetData(model, data) # reset again 
        b = randomise_ball(model, data, seed) # place with the same seed, save position as "b"

        for _ in range(500):                                # 500 steps x 0.002 s = 1 s of sim time
            mujoco.mj_step(model, data)                     # advance the physics one step
        settled = data.body("target_ball_body").xpos        # where the ball ended up (world coordinates)

        print(f"seed {seed}: start {a.round(3)}  same seed again: {np.allclose(a, b)}  "
              f"after 1 s: {settled.round(3)}")             # print start, whether a == b, and final position

    # visual check: new seed every 2 s
    with mujoco.viewer.launch_passive(model, data) as viewer:   # open the viewer window
        seed = 0                                                # start from seed 0
        while viewer.is_running():                              # loop until you close the window
            mujoco.mj_resetData(model, data)                    # reset the scene
            pos = randomise_ball(model, data, seed)             # place the ball for this seed
            print(f"showing seed {seed}: {pos.round(3)}")       # print which seed and where the ball is
            viewer.sync()                                       # push the new state to the viewer so it redraws
            time.sleep(2)                                       # wait 2 seconds
            seed += 1           