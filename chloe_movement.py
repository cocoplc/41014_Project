from pathlib import Path
import mujoco 
import mujoco.viewer

FOLDER = Path(__file__).parent

model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)