import tdgl
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon # For combining polygons

# -----------------------------------------------------------------------------
# 1. Define Superconducting Layer Properties
# -----------------------------------------------------------------------------
layer = tdgl.Layer(
    london_lambda=0.08,      # London penetration depth in um
    coherence_length=0.04,   # Ginzburg-Landau coherence length in um (xi)
    thickness=0.01,          # Film thickness in um
    conductivity=10,         # Normal state conductivity in Siemens/um
    u=5.79,                  # Ratio of order parameter relaxation times
    gamma=10                 # Inelastic scattering parameter
)

# -----------------------------------------------------------------------------
# 2. Define SQUID Geometry Parameters (in um)
# -----------------------------------------------------------------------------

# --- Choose SQUID Type ---
SQUID_TYPE = "circular"  # "circular" or "square"

# --- Loop Dimensions ---
if SQUID_TYPE == "circular":
    loop_outer_radius = 0.6
    loop_inner_radius = 0.4 # This defines the width of the SQUID arms
    loop_center = (0, 0)
elif SQUID_TYPE == "square":
    loop_outer_width = 1.2 # Full width of the outer square
    loop_inner_width = 0.8 # Full width of the inner square hole
    loop_center = (0, 0)

# --- Weak Link (Bridge/Junction) Dimensions ---
bridge_width = 0.05  # Width of the constriction (critical parameter)
bridge_length = 0.08 # Length of the constriction

# --- Terminal Dimensions (for current bias) ---
terminal_lead_width = 0.3
terminal_lead_length = 0.4 # Length of the lead extending from the SQUID
terminal_contact_width = terminal_lead_width
terminal_contact_length = 0.1 # How much the terminal polygon overlaps the lead

# -----------------------------------------------------------------------------
# 3. Construct the SQUID Film Polygon
# -----------------------------------------------------------------------------

# Create the main SQUID body (washer)
if SQUID_TYPE == "circular":
    outer_poly_points = tdgl.geometry.circle(radius=loop_outer_radius, points=200, center=loop_center)
    inner_poly_points = tdgl.geometry.circle(radius=loop_inner_radius, points=100, center=loop_center)
    device_name_suffix = f"circ_L{loop_outer_radius:.2f}_w{(loop_outer_radius-loop_inner_radius):.2f}"
elif SQUID_TYPE == "square":
    outer_poly_points = tdgl.geometry.box(width=loop_outer_width, height=loop_outer_width, points=200, center=loop_center)
    inner_poly_points = tdgl.geometry.box(width=loop_inner_width, height=loop_inner_width, points=100, center=loop_center)
    device_name_suffix = f"sq_L{loop_outer_width:.2f}_w{(loop_outer_width-loop_inner_width)/2:.2f}"

# Convert to tdgl.Polygon, then to shapely.Polygon for operations
outer_tdgl_poly = tdgl.Polygon(points=outer_poly_points)
# Give the hole a name so it can be referenced later for fluxoid calculations.
inner_tdgl_poly = tdgl.Polygon(name="squid_hole", points=inner_poly_points) # This is the hole
washer_shapely_poly = outer_tdgl_poly.polygon.difference(inner_tdgl_poly.polygon)

# Define the bridge polygons
# Bridge 1 (right side)
br1_center_y = 0
if SQUID_TYPE == "circular":
    br1_center_x = loop_inner_radius + bridge_length / 2
elif SQUID_TYPE == "square":
    br1_center_x = loop_inner_width / 2 + bridge_length / 2
bridge1_points = tdgl.geometry.box(width=bridge_length, height=bridge_width, center=(br1_center_x, br1_center_y))
bridge1_tdgl_poly = tdgl.Polygon(points=bridge1_points)

# Bridge 2 (left side)
br2_center_y = 0
if SQUID_TYPE == "circular":
    br2_center_x = -loop_inner_radius - bridge_length / 2
elif SQUID_TYPE == "square":
    br2_center_x = -loop_inner_width / 2 - bridge_length / 2
bridge2_points = tdgl.geometry.box(width=bridge_length, height=bridge_width, center=(br2_center_x, br2_center_y))
bridge2_tdgl_poly = tdgl.Polygon(points=bridge2_points)

# Create cutouts in the washer where the bridges will be.
cutout_width_factor = 1.05
cutout1_points = tdgl.geometry.box(width=bridge_length, height=bridge_width * cutout_width_factor, center=(br1_center_x, br1_center_y))
cutout1_shapely_poly = tdgl.Polygon(points=cutout1_points).polygon
cutout2_points = tdgl.geometry.box(width=bridge_length, height=bridge_width * cutout_width_factor, center=(br2_center_x, br2_center_y))
cutout2_shapely_poly = tdgl.Polygon(points=cutout2_points).polygon

# Subtract cutouts from washer, then add bridges
film_with_gaps_shapely = washer_shapely_poly.difference(cutout1_shapely_poly).difference(cutout2_shapely_poly)
final_film_shapely_poly = film_with_gaps_shapely.union(bridge1_tdgl_poly.polygon).union(bridge2_tdgl_poly.polygon)

# Define leads for terminals
# Lead 1 (top)
lead1_center_x = 0
if SQUID_TYPE == "circular":
    lead1_center_y = loop_outer_radius + terminal_lead_length / 2
elif SQUID_TYPE == "square":
    lead1_center_y = loop_outer_width / 2 + terminal_lead_length / 2
lead1_points = tdgl.geometry.box(width=terminal_lead_width, height=terminal_lead_length, center=(lead1_center_x, lead1_center_y))
lead1_tdgl_poly = tdgl.Polygon(points=lead1_points)

# Lead 2 (bottom)
lead2_center_x = 0
if SQUID_TYPE == "circular":
    lead2_center_y = -loop_outer_radius - terminal_lead_length / 2
elif SQUID_TYPE == "square":
    lead2_center_y = -loop_outer_width / 2 - terminal_lead_length / 2
lead2_points = tdgl.geometry.box(width=terminal_lead_width, height=terminal_lead_length, center=(lead2_center_x, lead2_center_y))
lead2_tdgl_poly = tdgl.Polygon(points=lead2_points)

# Union film with leads
final_film_shapely_poly = final_film_shapely_poly.union(lead1_tdgl_poly.polygon).union(lead2_tdgl_poly.polygon)

# **THE FIX IS HERE:** Extract the exterior coordinates from the final shapely object
# and pass them to the 'points' argument.
film_polygon = tdgl.Polygon(
    name="squid_film",
    points=list(final_film_shapely_poly.exterior.coords)
)
device_name_suffix += f"_jw{bridge_width:.3f}_jl{bridge_length:.3f}".replace(".","p")

# The main hole of the SQUID is the named inner_tdgl_poly.
holes_list = [inner_tdgl_poly]

# --- Define Terminals ---
# Source terminal (top)
source_center_y = lead1_center_y + terminal_lead_length / 2 - terminal_contact_length / 2
source_points = tdgl.geometry.box(width=terminal_contact_width, height=terminal_contact_length, center=(lead1_center_x, source_center_y))
source_terminal = tdgl.Polygon(name="source", points=source_points)

# Drain terminal (bottom)
drain_center_y = lead2_center_y - terminal_lead_length / 2 + terminal_contact_length / 2
drain_points = tdgl.geometry.box(width=terminal_contact_width, height=terminal_contact_length, center=(lead2_center_x, drain_center_y))
drain_terminal = tdgl.Polygon(name="drain", points=drain_points)

terminals_list = [source_terminal, drain_terminal]

# -----------------------------------------------------------------------------
# 4. Create the TDGL Device
# -----------------------------------------------------------------------------
device = tdgl.Device(
    name=f"dc_squid_{device_name_suffix}",
    layer=layer,
    film=film_polygon,
    holes=holes_list,
    terminals=terminals_list,
    length_units="um"
)

# -----------------------------------------------------------------------------
# 5. Generate the Mesh
# -----------------------------------------------------------------------------
mesh_edge_length = min(layer.coherence_length / 2.5, bridge_width / 2.5)
print(f"Attempting to mesh '{device.name}' with max_edge_length = {mesh_edge_length:.4f} um...")
device.make_mesh(max_edge_length=mesh_edge_length, min_points=3000)
print(f"Mesh generated with {len(device.mesh.sites)} sites and {len(device.mesh.elements)} elements.")

device.plot(mesh=True, legend=True)
plt.title(f"Device: {device.name} with Mesh")
plt.show()

# -----------------------------------------------------------------------------
# 6. Define Solver Options
# -----------------------------------------------------------------------------
solver_options = tdgl.SolverOptions(
    solve_time=300,
    dt_init=1e-4,
    dt_max=0.05,
    save_every=150,
    output_file=f"{device.name}_results.h5",
    field_units="mT",
    current_units="uA",
    terminal_psi=0 # Normal metal contacts
)

# -----------------------------------------------------------------------------
# 7. Define Applied Magnetic Field and Terminal Currents
# -----------------------------------------------------------------------------
applied_magnetic_field_strength = 0.01  # mT
if applied_magnetic_field_strength != 0:
    applied_vector_potential = tdgl.sources.ConstantField(
        value=applied_magnetic_field_strength,
        field_units=solver_options.field_units,
        length_units=device.length_units
    )
else:
    applied_vector_potential = 0

# Terminal currents
applied_current = 0.5  # uA
if terminals_list:
    terminal_currents_dict = {
        "source": applied_current,
        "drain": -applied_current
    }
else:
    terminal_currents_dict = None

# -----------------------------------------------------------------------------
# 8. Run the Simulation
# -----------------------------------------------------------------------------
print(f"Starting TDGL simulation for {device.name} with B_applied = {applied_magnetic_field_strength} mT, I_bias = {applied_current if terminals_list else 0} uA...")
solution = tdgl.solve(
    device,
    solver_options,
    applied_vector_potential=applied_vector_potential,
    terminal_currents=terminal_currents_dict,
    disorder_epsilon=1
)
print("TDGL simulation finished.")

# -----------------------------------------------------------------------------
# 9. Analyze and Visualize Results
# -----------------------------------------------------------------------------
if solution is not None:
    print(f"Solution data saved to: {solution.path}")
    solution.solve_step = -1

    fig_psi, ax_psi = solution.plot_order_parameter(squared=False)
    title_psi = (f"Order Parameter |ψ| in {device.name}\n"
                 f"Junctions: W={bridge_width:.2f}um, L={bridge_length:.2f}um. "
                 f"B={applied_magnetic_field_strength}mT, I={applied_current if terminals_list else 0}{solver_options.current_units}\n"
                 f"t={solution.tdgl_data.state['time']:.2f} τ₀")
    fig_psi.suptitle(title_psi)
    plt.show()

    fig_K, ax_K = solution.plot_currents(streamplot=True, cmap="inferno", auto_range_cutoff=1)
    title_K = (f"Total Current Density in {device.name}\n"
               f"Junctions: W={bridge_width:.2f}um, L={bridge_length:.2f}um. "
               f"B={applied_magnetic_field_strength}mT, I={applied_current if terminals_list else 0}{solver_options.current_units}\n"
               f"t={solution.tdgl_data.state['time']:.2f} τ₀")
    fig_K.suptitle(title_K)
    plt.show()
    
    # Calculate and print fluxoid in the SQUID loop
    if holes_list:
        hole_to_analyze = holes_list[0]
        try:
            fluxoid_in_loop = solution.hole_fluxoid(hole_name=hole_to_analyze.name)
            print(f"\nFluxoid in SQUID loop ('{hole_to_analyze.name}'):")
            print(f"  Total Fluxoid: {fluxoid_in_loop.total_fluxoid:.3f} Phi_0")
            print(f"  Flux Part (Integral A.dl): {fluxoid_in_loop.flux_part:.3f} Phi_0")
            print(f"  Supercurrent Part (Integral Lambda*J.dl): {fluxoid_in_loop.supercurrent_part:.3f} Phi_0")
        except Exception as e:
            print(f"Could not calculate fluxoid for hole '{hole_to_analyze.name}': {e}")

else:
    print("Simulation did not produce a solution.")
