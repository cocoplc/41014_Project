import mujoco
import numpy as np

TABLE_CENTRE = np.array([0.04, -0.29])
BALL_Z = 0.145  # table top 0.12 + radius 0.025

# function for randomising ball 
def randomise_ball(model, data, seed):
    rng = np.random.default_rng(seed)
    x = TABLE_CENTRE[0] + rng.uniform(-0.10, 0.10)
    y = TABLE_CENTRE[1] + rng.uniform(-0.08, 0.08)
    adr = model.jnt_qposadr[model.joint("target_ball_freejoint").id]
    data.qpos[adr:adr+3] = [x, y, BALL_Z]
    data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    return np.array([x, y, BALL_Z])


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    for seed in range(5):
        mujoco.mj_resetData(model, data)
        a = randomise_ball(model, data, seed)
        mujoco.mj_resetData(model, data)
        b = randomise_ball(model, data, seed)

        for _ in range(500):  # 1 s of sim time
            mujoco.mj_step(model, data)
        settled = data.body("target_ball_body").xpos

        print(f"seed {seed}: start {a.round(3)}  same seed again: {np.allclose(a, b)}  "
              f"after 1 s: {settled.round(3)}")