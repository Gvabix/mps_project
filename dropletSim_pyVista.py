# %% [markdown]
# # Optimized 3D PyVista Cloud Simulation
# Object-oriented implementation for simulating and visualizing droplet dynamics.

# %%
import os
import sys
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional

os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

if 'google.colab' in sys.modules:
    import pyvista as pv
    # xvfb is required for headless rendering in Colab
    if not os.path.exists('/usr/bin/Xvfb'):
        os.system('apt-get install -qq xvfb')
    pv.start_xvfb()
else:
    import pyvista as pv

from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.utils.kinematic_2d import Storage, Simulation
import PySDM.products as PySDM_products
from PySDM import Formulae
from PySDM.physics import si

class DropletSimulation:
    """
    A class to handle 3D droplet simulation using PySDM and visualization using PyVista.
    """

    def __init__(
        self, 
        grid_size: tuple = (24, 24), 
        n_sd_per_gridbox: int = 16, 
        simulation_time: float = 64 * si.minute, 
        dt: float = 1.0 * si.second
    ):
        """
        Initializes the simulation settings and PySDM objects.

        Args:
            grid_size: Tuple representing the number of grid boxes (nx, nz).
            n_sd_per_gridbox: Number of super-droplets per grid box.
            simulation_time: Total time of the simulation in seconds.
            dt: Time step in seconds.
        """
        self.settings = Settings(Formulae())
        self.settings.grid = grid_size
        self.settings.n_sd_per_gridbox = n_sd_per_gridbox
        self.settings.simulation_time = simulation_time
        self.settings.dt = dt
        
        # Calculate output steps based on simulation time and dt
        self.output_steps = np.arange(0, self.settings.simulation_time + self.settings.dt, self.settings.dt).astype(int)
        
        self.tracked_products = [PySDM_products.EffectiveRadius(unit='um')]
        self.storage = Storage()
        self.sim = Simulation(self.settings, self.storage, SpinUp)
        self.sim.reinit(products=self.tracked_products)
        
        self.product_key = list(self.sim.particulator.products.keys())[0]
        
        # Cache attribute keys for faster access during simulation
        attr_dict = self.sim.particulator.attributes._ParticleAttributes__attributes
        self.attr_keys = {
            'pos': [k for k in attr_dict.keys() if 'pos' in k.lower()][0],
            'rad': [k for k in attr_dict.keys() if 'rad' in k.lower() or 'size' in k.lower()][0]
        }
        
        self.spatial_frames: List[Dict[str, np.ndarray]] = []

    def run(self) -> List[Dict[str, np.ndarray]]:
        """
        Runs the physics engine and stores spatial frames for visualization.

        Returns:
            A list of dictionaries, each containing grid data, particle coordinates, and radii.
        """
        self.spatial_frames = []
        attr_dict = self.sim.particulator.attributes._ParticleAttributes__attributes
        
        for step in tqdm(self.output_steps, desc="Running Physics Engine"):
            self.sim.particulator.run(step - self.sim.particulator.n_steps)
            
            # Extract grid data
            grid_data = self.sim.particulator.products[self.product_key].get().copy()
            
            # Extract particle attributes
            pos_obj = attr_dict[self.attr_keys['pos']]
            rad_obj = attr_dict[self.attr_keys['rad']]
            
            pos_data = pos_obj.get() if hasattr(pos_obj, 'get') else pos_obj.storage
            rad_data = rad_obj.get() if hasattr(rad_obj, 'get') else rad_obj.storage
            
            raw_pos = pos_data.ndarray.copy() if hasattr(pos_data, 'ndarray') else np.asarray(pos_data).copy()
            raw_r = rad_data.ndarray.copy() if hasattr(rad_data, 'ndarray') else np.asarray(rad_data).copy()
            
            # Unit conversion if necessary (m to um)
            if np.max(raw_r) < 1e-2: 
                raw_r = raw_r * 1e6
                
            # Handle coordinate mapping
            if raw_pos.shape[0] == 2:
                raw_x, raw_z = raw_pos[0], raw_pos[1]
            else:
                raw_x, raw_z = raw_pos[:, 0], raw_pos[:, 1]
                
            raw_r = raw_r.flatten()
            
            # Map coordinates to 3D space for PyVista (Z -> X, X -> Z)
            particle_coords = np.zeros((len(raw_x), 3))
            particle_coords[:, 0] = raw_z * self.settings.size[0]
            particle_coords[:, 1] = 0.1
            particle_coords[:, 2] = raw_x * self.settings.size[1]
            
            self.spatial_frames.append({
                'grid': grid_data,
                'coords': particle_coords,
                'radii': raw_r
            })
            
        return self.spatial_frames

    def generate_gif(self, filename: str = 'droplet_simulation.gif', fps: int = 10, render_every: int = 1) -> None:
        """
        Renders the simulation results and saves them as a GIF.

        Args:
            filename: The name of the output GIF file.
            fps: Frames per second for the animation.
            render_every: Number of steps to skip between rendered frames.
        """
        if not self.spatial_frames:
            print("No simulation data found. Please run the simulation first.")
            return

        # Pre-calculate color limits for consistent colormap across all frames
        grid_vals = np.hstack([f['grid'].T.flatten() for f in self.spatial_frames]).astype(float)
        p2, p98 = np.nanpercentile(grid_vals, [2, 98])
        
        if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
            grid_min, grid_max = np.nanmin(grid_vals), np.nanmax(grid_vals)
            if grid_min == grid_max:
                grid_max += 1.0
        else:
            grid_min, grid_max = float(p2), float(p98)

        # Setup Plotter
        plotter = pv.Plotter(notebook=True, off_screen=True, window_size=(1200, 800))
        plotter.set_background('white')
        plotter.disable_anti_aliasing()
        plotter.enable_depth_peeling()

        # Initialize Background Grid Mesh
        grid_mesh = pv.ImageData(dimensions=(self.settings.grid[0] + 1, 1, self.settings.grid[1] + 1))
        grid_mesh.spacing = (self.settings.size[0] / self.settings.grid[0], 1, self.settings.size[1] / self.settings.grid[1])
        grid_mesh.cell_data["Grid Radius"] = np.array(self.spatial_frames[0]['grid'].T.flatten(order="C"), dtype=float)

        plotter.add_mesh(
            grid_mesh,
            scalars="Grid Radius",
            cmap="viridis",
            clim=[grid_min, grid_max],
            opacity=0.5,
            show_scalar_bar=True,
            lighting=False,
            scalar_bar_args={
                "title": "Grid Radius [um]",
                "color": "black",
                "position_x": 0.02,
                "position_y": 0.12,
                "width": 0.08,
                "height": 0.18,
                "title_font_size": 10,
                "label_font_size": 8,
            },
        )

        # Initialize Point Cloud and Glyphs
        init_frame = self.spatial_frames[0]
        point_cloud = pv.PolyData(init_frame['coords'])
        point_cloud["Display Size"] = init_frame['radii'] * 0.3 + 2.0
        
        droplet_geom = pv.Sphere(phi_resolution=16, theta_resolution=16)
        droplet_glyphs = point_cloud.glyph(scale="Display Size", factor=1.0, geom=droplet_geom)

        particle_actor = plotter.add_mesh(
            droplet_glyphs,
            color="red",
            lighting=False,
            ambient=1.0,
        )

        # Domain boundary
        domain_box = pv.Box(bounds=(0, self.settings.size[0], 0, 0.2, 0, self.settings.size[1]))
        plotter.add_mesh(domain_box, color="black", style="wireframe", opacity=0.2)

        # Camera Configuration
        plotter.view_xz()
        plotter.camera.SetParallelProjection(True)
        plotter.reset_camera()

        time_text = plotter.add_text("t = 0.0 min", position=(10, 760), font_size=14, color="black")

        # Compile Animation
        plotter.open_gif(filename, fps=fps)

        for step, frame in zip(self.output_steps, tqdm(self.spatial_frames, desc="Generating GIF")):
            if step % render_every != 0:
                continue

            grid_mesh.cell_data["Grid Radius"] = np.array(frame['grid'].T.flatten(order="C"), dtype=float)
            
            point_cloud.points = frame['coords']
            point_cloud["Display Size"] = frame['radii'] * 0.3 + 2.0
            
            updated_glyphs = point_cloud.glyph(scale="Display Size", factor=1.0, geom=droplet_geom)
            particle_actor.mapper.dataset = updated_glyphs
            
            time_text.SetInput(f"t = {step / si.minute:.1f} min")
            
            plotter.render()
            plotter.write_frame()

        plotter.close()
        print(f"Animation saved to {filename}")

# %% [markdown]
# ## Execution Example

# %%
if __name__ == "__main__":
    # Create simulation instance with custom parameters
    sim_runner = DropletSimulation(
        grid_size=(16, 16),
        n_sd_per_gridbox = 8,
        simulation_time = 32 * si.minute,
        dt = 5.0 * si.second
    )
    
    # Run simulation
    sim_runner.run()
    
    # Generate visualization
    sim_runner.generate_gif(filename='cloud_simulation_refactored.gif', fps=10, render_every=12)

    # Display in notebook if applicable
    try:
        from IPython.display import Image, display
        display(Image(filename='cloud_simulation_refactored.gif'))
    except ImportError:
        pass
