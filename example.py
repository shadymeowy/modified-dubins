from modified_dubins import dubins, generate_modified_path

import numpy as np
import matplotlib.pyplot as plt

# Set constraints and start and goal points/attitudes
# They should be in the form of a dictionary
# so that Cython can automatically convert them to C
# structs
args = {
    "e_z": np.array([0., 0., 1.]),
    "position": np.array([0., -8., 3.]),
    "direction": np.array([4., 2., 3.]),
    "target_position": np.array([8., 0., 4.]),
    "target_direction": np.array([2., -3., 1.]),
    "r": 3,
}
# Generate the path parameters
paths = dubins(args)

# Because C code returns an array of structs,
# whose some elements may be not used/initialized,
paths = paths["paths"][:paths["count"]]

# Sort the paths by the length of the path
paths.sort(key=lambda x: x["cost"])

# Maximum vertical acceleration
vaccel = 0.2

# Calculate and generate the modified path
# Originals paths are guaranteed to be valid
# However, modified paths may not be valid
# See generate_path
for path in paths:
    t = np.linspace(0, path["cost"])
    out = np.zeros((t.shape[0], 3))
    if generate_modified_path(t, out, path, args, vaccel):
        break

# Set up the 3D plot
ax = plt.figure().add_subplot(111, projection='3d')
ax.plot(out[:, 0], out[:, 1], out[:, 2], label="path")

# Normalize the direction vectors to plot
args["direction"] /= np.linalg.norm(args["direction"])
args["target_direction"] /= np.linalg.norm(args["target_direction"])

# Plot the start and goal points and attitudes
ax.plot(
    *np.transpose(np.vstack((
        args["position"],
        args["position"] + args["direction"]))
    ),
    label="start attitude"
)
ax.plot(
    *np.transpose(np.vstack((
        args["target_position"],
        args["target_position"] + args["target_direction"]))
    ),
    label="target attitude"
)

# One liner trick to make the axis have equal scale
# See https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

ax.legend()
plt.show()