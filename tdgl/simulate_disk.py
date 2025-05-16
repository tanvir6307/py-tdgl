import tdgl
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Define Superconducting Layer Properties
# -----------------------------------------------------------------------------
# All length units are in micrometers (um) unless otherwise specified by pint later.
# Other units are derived from the base Ginzburg-Landau dimensionless units.
layer = tdgl.Layer(
    london_lambda=0.08,      # London penetration depth in um
    coherence_length=0.04,   # Ginzburg-Landau coherence length in um
    thickness=0.01,          # Film thickness in um
    conductivity=10,         # Normal state conductivity in Siemens/um (typical value)
    u=5.79,                  # Ratio of order parameter relaxation times (standard for dirty limit)
    gamma=10                 # Inelastic scattering parameter (standard for dirty limit)
)

# -----------------------------------------------------------------------------
# 2. Define the Device Geometry: A simple disk
# -----------------------------------------------------------------------------
# Points defining the superconducting disk film
# tdgl.geometry.circle(radius, number_of_points_on_perimeter, center_coordinates)
disk_radius = 0.5  # um
film_points = tdgl.geometry.circle(radius=disk_radius, points=200, center=(0, 0))
film_polygon = tdgl.Polygon(name="disk_film", points=film_points)

# No holes or terminals for this simple disk simulation
holes_list = []
terminals_list = []

# -----------------------------------------------------------------------------
# 3. Create the TDGL Device
# -----------------------------------------------------------------------------
device = tdgl.Device(
    name="simple_disk",
    layer=layer,
    film=film_polygon,
    holes=holes_list,
    terminals=terminals_list,
    length_units="um"  # Specify the units used for defining coordinates and layer params
)

# -----------------------------------------------------------------------------
# 4. Generate the Mesh for Finite Volume Method
# -----------------------------------------------------------------------------
# max_edge_length should generally be smaller than the coherence length
# to resolve variations in the order parameter.
# A smaller value gives a finer mesh but increases computation time.
mesh_edge_length = layer.coherence_length / 2.0
print(f"Attempting to mesh with max_edge_length = {mesh_edge_length:.3f} um...")
device.make_mesh(max_edge_length=mesh_edge_length, min_points=1000) # min_points ensures a reasonable number of mesh sites
print(f"Mesh generated with {len(device.mesh.sites)} sites and {len(device.mesh.elements)} elements.")

# Optional: Plot the device and its mesh to verify
# device.plot(mesh=True, legend=False)
# plt.title("Disk Device with Mesh")
# plt.show()

# -----------------------------------------------------------------------------
# 5. Define Solver Options
# -----------------------------------------------------------------------------
solver_options = tdgl.SolverOptions(
    solve_time=200,              # Total dimensionless time for the simulation
    dt_init=1e-3,                # Initial dimensionless time step
    dt_max=0.1,                  # Maximum adaptive time step
    save_every=100,              # Save data every 100 dimensionless time steps
    output_file="disk_simulation_results.h5", # Name of the HDF5 output file
    field_units="mT",            # Units for applied magnetic field
    current_units="uA"           # Units for any applied currents (not used here)
)

# -----------------------------------------------------------------------------
# 6. Define Applied Magnetic Field (Optional, but interesting)
# -----------------------------------------------------------------------------
# Let's apply a small, uniform out-of-plane magnetic field.
# The value is in 'mT' as specified in solver_options.field_units.
applied_magnetic_field_strength = 0.1  # mT
# tdgl.sources.ConstantField creates a Parameter object for a uniform field.
# It calculates the appropriate vector potential for this field.
applied_vector_potential = tdgl.sources.ConstantField(
    value=applied_magnetic_field_strength,
    field_units=solver_options.field_units,
    length_units=device.length_units
)
# If no field is desired, set: applied_vector_potential = 0

# -----------------------------------------------------------------------------
# 7. Run the Simulation
# -----------------------------------------------------------------------------
print("Starting TDGL simulation...")
# The disorder_epsilon=1 means a homogeneous superconductor (Tc is uniform).
# No terminal currents are applied in this case.
solution = tdgl.solve(
    device,
    solver_options,
    applied_vector_potential=applied_vector_potential,
    terminal_currents=None,
    disorder_epsilon=1,
    seed_solution=None # No initial seed, starts from psi=1, mu=0
)
print("TDGL simulation finished.")

# -----------------------------------------------------------------------------
# 8. Analyze and Visualize Results (Example: Plot final order parameter)
# -----------------------------------------------------------------------------
if solution is not None:
    print(f"Solution data saved to: {solution.path}")
    # Load the final state of the simulation
    solution.solve_step = -1 # -1 loads the last saved step

    # Plot the magnitude of the order parameter
    fig, ax = solution.plot_order_parameter(squared=False) # squared=True plots |psi|^2
    fig.suptitle(f"Order Parameter Magnitude |ψ| at t={solution.tdgl_data.state['time']:.2f} τ₀, B_applied={applied_magnetic_field_strength} mT")
    plt.show()

    # You can also plot other quantities, e.g.:
    # fig, ax = solution.plot_currents(dataset="supercurrent", streamplot=True)
    # fig.suptitle(f"Supercurrent Density at t={solution.tdgl_data.state['time']:.2f} τ₀")
    # plt.show()

else:
    print("Simulation did not produce a solution (e.g., cancelled during thermalization).")

