"""Standalone geo/map plotting helpers."""

from __future__ import annotations

import io
import math
import os
import shutil as _shutil
import subprocess
import urllib.request

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyproj import Transformer

def plot_surface_on_map(self,
                         mapa,
                         time=0.0,
                         component='resultant',
                         data_type='vel',
                         factor=1,
                         cmap='RdBu_r',
                         thresh_pct=0.01,
                         radius=3,
                         fill_opacity=0.85,
                         crs_utm='EPSG:32719'):
    """Overlay a single time snapshot on an existing Folium map.

    Converts node coordinates from UTM to lon/lat and adds one
    CircleMarker per active node (|mag| >= thresh_pct * vmax).
    The caller is responsible for displaying or saving the map.

    Parameters
    ----------
    mapa : folium.Map
        Pre-built map (tiles, station markers, etc. already added).
    time : float
        Simulation time [s] to plot.
    component : {'z', 'e', 'n', 'resultant'}
        Signal component.
    data_type : {'vel', 'accel', 'disp'}
    cmap : str
        Matplotlib colormap name.
    thresh_pct : float
        Fraction of vmax below which nodes are not plotted (0.01 = 1%).
    radius : int
        CircleMarker radius in pixels.
    fill_opacity : float
    crs_utm : str
        EPSG code of self.xyz coordinates.
        Default 'EPSG:32719' (UTM zone 19S, central Chile).

    Returns
    -------
    mapa : folium.Map
        Same map object with data markers added.
    """
    import folium
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from pyproj import Transformer

    # Colour limits
    if self._vmax is None:
        self._compute_vmax()
    if component.lower() == 'resultant':
        vmax = self._vmax[data_type]['resultant']
        vmin = 0.0
    else:
        vmax = self._vmax[data_type][component.lower()]
        vmin = -vmax

    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm    = plt.get_cmap(cmap)

    # Snapshot
    it = int(np.argmin(np.abs(self.time - time)))
    if component.lower() == 'resultant':
        mag = np.sqrt(
            self.get_surface_snapshot(it, 'e', data_type) ** 2 +
            self.get_surface_snapshot(it, 'n', data_type) ** 2 +
            self.get_surface_snapshot(it, 'z', data_type) ** 2)
    else:
        mag = self.get_surface_snapshot(it, component, data_type)
    mag= mag*factor
    # Coordinates: xyz[:,0]=Northing, xyz[:,1]=Easting (km → m)
    transformer = Transformer.from_crs(crs_utm, 'EPSG:4326', always_xy=True)
    lons, lats  = transformer.transform(
        self.xyz[:, 1] * 1000.0,   # Easting
        self.xyz[:, 0] * 1000.0)   # Northing

    # Active nodes only
    thresh = vmax * thresh_pct
    active = np.abs(mag) >= thresh

    # Helper: magnitude → hex colour
    def _to_hex(val):
        r, g, b, _ = cm(norm(val))
        return '#{:02x}{:02x}{:02x}'.format(
            int(r * 255), int(g * 255), int(b * 255))

    # Add markers
    for lon, lat, m in zip(lons[active], lats[active], mag[active]):
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=_to_hex(m),
            fill=True,
            fill_color=_to_hex(m),
            fill_opacity=fill_opacity,
            weight=0,
        ).add_to(mapa)

    print(f"t = {self.time[it]:.3f}s | active nodes: {active.sum()}/{len(mag)}")
    return mapa


def create_animation_map(self,
                          time_start=0.0,
                          time_end=None,
                          n_frames=50,
                          component='resultant',
                          data_type='vel',
                          factor=1.0,
                          cmap='RdBu_r',
                          thresh_pct=0.01,
                          radius=4,
                          fill_opacity=0.85,
                          figsize=(14, 10),
                          dpi=100,
                          fps=10,
                          ffmpeg_path=None,
                          output_dir='animation_map',
                          output_video='animation_map.mp4',
                          crs_utm='EPSG:32719',
                          tile_zoom=11,
                          tile_url='https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                          stations=None):
    """Render a map animation overlaying node data on a tile basemap.

    Downloads basemap tiles once using urllib (no extra dependencies),
    stitches them into a mosaic, then renders one matplotlib frame per
    time step with the data scatter on top.  Frames are assembled into
    a video with ffmpeg.

    Parameters
    ----------
    time_start : float
        Start time [s].
    time_end : float, optional
        End time [s]. Defaults to last available time step.
    n_frames : int
        Total number of frames in the animation.
    component : {'z', 'e', 'n', 'resultant'}
        Signal component to plot.
    data_type : {'vel', 'accel', 'disp'}
    factor : float
        Scale factor applied to the data before plotting.
        Default 1.0. Use e.g. 100.0 to convert m/s → cm/s.
    cmap : str
        Matplotlib colormap name.
    thresh_pct : float
        Nodes with |mag| < thresh_pct * vmax are not plotted.
        0.01 = only nodes above 1 % of the global maximum.
    radius : int
        Scatter marker size in points².
    fill_opacity : float
        Marker transparency (0–1).
    figsize : tuple of float
        Figure size in inches (width, height).
    dpi : int
        Frame resolution.
    fps : int
        Frames per second of the output video.
    ffmpeg_path : str, optional
        Full path to ffmpeg binary. If None, uses shutil.which('ffmpeg').
    output_dir : str
        Directory where frame PNGs are saved.
    output_video : str
        Output .mp4 file path.
    crs_utm : str
        EPSG code of self.xyz coordinates.
        Default 'EPSG:32719' (UTM zone 19S, central Chile).
    tile_zoom : int
        Slippy map zoom level (10–13 recommended).
        Higher = more detail but more tiles to download.
    tile_url : str
        Tile URL template with {z}, {x}, {y} placeholders.
        Default: CartoDB Positron (clean, light basemap).
        Alternatives:
          OSM:  'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'
    stations : dict, optional
        Station markers drawn on every frame. Keys:

        - ``utmx``   : np.ndarray — Easting  [m]
        - ``utmy``   : np.ndarray — Northing [m]
        - ``names``  : list of str — station labels
        - ``colors`` : list of str — one matplotlib color per station
        - ``marker`` : str — matplotlib marker symbol (default '^')
        - ``size``   : int — marker size in points² (default 60)

        Example::

            color_map = {
                'red'      : '#d73027', 'blue'     : '#1a6faf',
                'green'    : '#2ca25f', 'purple'   : '#7b2d8b',
                'orange'   : '#f46d43', 'darkred'  : '#8b0000',
                'lightred' : '#fc8d59', 'beige'    : '#f5f5dc',
                'darkblue' : '#003399', 'darkgreen': '#1a5c1a',
                'cadetblue': '#5f9ea0',
            }
            stations = {
                'utmx'  : utmx,
                'utmy'  : utmy,
                'names' : utm_order,
                'colors': [color_map[c] for c in colors],
                'marker': '^',
                'size'  : 60,
            }
    """
    import os
    import math
    import io
    import subprocess
    import shutil as _shutil
    import urllib.request

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from PIL import Image
    from pyproj import Transformer

    #  colour limits (fixed for all frames) 
    if self._vmax is None:
        self._compute_vmax()
    if time_end is None:
        time_end = self.time[-1]

    # Apply ``factor`` exactly once to the cached vmax. Historical note:
    # the previous version multiplied by ``factor`` here and then again on
    # the next line, which silently squared the user's scale. ``vmin``
    # stays asymmetric for resultant (always 0) and signed otherwise.
    if component.lower() == 'resultant':
        vmax = self._vmax[data_type]['resultant'] * factor
        vmin = 0.0
    else:
        vmax = self._vmax[data_type][component.lower()] * factor
        vmin = -vmax
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    thresh = vmax * thresh_pct

    comp_label  = {'z': 'Vertical (Z)', 'e': 'East (E)',
                   'n': 'North (N)', 'resultant': 'Resultant'}[component.lower()]
    dtype_label = {'vel': 'Velocity', 'accel': 'Acceleration',
                   'disp': 'Displacement'}[data_type]

    #  node coordinates (computed once) 
    # self.xyz[:,0] = Northing, self.xyz[:,1] = Easting (km → m)
    transformer = Transformer.from_crs(crs_utm, 'EPSG:4326', always_xy=True)
    lons, lats  = transformer.transform(
        self.xyz[:, 1] * 1000.0,
        self.xyz[:, 0] * 1000.0)

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    pad_lon = (lon_max - lon_min) * 0.08
    pad_lat = (lat_max - lat_min) * 0.08
    plot_extent = [lon_min - pad_lon, lon_max + pad_lon,
                   lat_min - pad_lat, lat_max + pad_lat]

    #  station coordinates (computed once) 
    sta_lons = sta_lats = sta_names = sta_colors = None
    if stations is not None:
        sta_lons, sta_lats = transformer.transform(
            stations['utmx'],
            stations['utmy'])
        sta_names  = stations.get('names',  ['']*len(sta_lons))
        sta_colors = stations.get('colors', ['red']*len(sta_lons))
        sta_marker = stations.get('marker', '^')
        sta_size   = stations.get('size',   60)

    #  tile helpers 
    def _deg2tile(lat_deg, lon_deg, z):
        n     = 2 ** z
        x     = int((lon_deg + 180.0) / 360.0 * n)
        lat_r = math.radians(lat_deg)
        y     = int((1.0 - math.log(
            math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
        return x, y

    def _tile2lon(x, z):
        return x / 2**z * 360.0 - 180.0

    def _tile2lat(y, z):
        return math.degrees(math.atan(
            math.sinh(math.pi * (1.0 - 2.0 * y / 2**z))))

    #  download and stitch basemap tiles (once) 
    print(f"Downloading basemap tiles (zoom={tile_zoom})...")
    headers = {'User-Agent': 'ShakerMaker/1.0 (research)'}

    x0, y0 = _deg2tile(lat_max + pad_lat, lon_min - pad_lon, tile_zoom)
    x1, y1 = _deg2tile(lat_min - pad_lat, lon_max + pad_lon, tile_zoom)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    print(f"  Fetching {n_tiles} tiles ({x1-x0+1} x {y1-y0+1})...")

    mosaic = Image.new('RGB', ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            url = tile_url.format(z=tile_zoom, x=tx, y=ty)
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    tile = Image.open(io.BytesIO(resp.read())).convert('RGB')
            except Exception:
                tile = Image.new('RGB', (256, 256), color=(230, 230, 230))
            mosaic.paste(tile, ((tx - x0) * 256, (ty - y0) * 256))

    mosaic_extent = [
        _tile2lon(x0,     tile_zoom),   # west
        _tile2lon(x1 + 1, tile_zoom),   # east
        _tile2lat(y1 + 1, tile_zoom),   # south
        _tile2lat(y0,     tile_zoom),   # north
    ]
    basemap = np.array(mosaic)
    print(f"  Basemap ready: {basemap.shape}  "
          f"lon=[{mosaic_extent[0]:.3f}, {mosaic_extent[1]:.3f}]  "
          f"lat=[{mosaic_extent[2]:.3f}, {mosaic_extent[3]:.3f}]")

    #  frame loop 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Rendering {n_frames} frames → {output_dir}/")

    for i, t_frame in enumerate(np.linspace(time_start, time_end, n_frames)):
        it = int(np.argmin(np.abs(self.time - t_frame)))

        # Snapshot
        if component.lower() == 'resultant':
            mag = np.sqrt(
                self.get_surface_snapshot(it, 'e', data_type) ** 2 +
                self.get_surface_snapshot(it, 'n', data_type) ** 2 +
                self.get_surface_snapshot(it, 'z', data_type) ** 2)
        else:
            mag = self.get_surface_snapshot(it, component, data_type)

        mag    = mag * factor
        active = np.abs(mag) >= thresh

        # Build frame
        fig, ax = plt.subplots(figsize=figsize)

        # Basemap
        ax.imshow(basemap, extent=mosaic_extent,
                  aspect='auto', origin='upper', zorder=0)

        # Faint background dots for all nodes
        ax.scatter(lons, lats, c='gray', s=1, alpha=0.08,
                   linewidths=0, zorder=1)

        # Data scatter
        if active.any():
            ax.scatter(lons[active], lats[active],
                       c=mag[active], cmap=cmap, norm=norm,
                       s=radius, alpha=fill_opacity,
                       linewidths=0, zorder=2)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02,
                     label=f'{dtype_label} [{comp_label}]')

        # Station markers (same on every frame)
        if sta_lons is not None:
            for lon, lat, name, color in zip(
                    sta_lons, sta_lats, sta_names, sta_colors):
                ax.scatter(lon, lat,
                           marker=sta_marker, color=color,
                           s=sta_size, zorder=5,
                           edgecolors='white', linewidths=0.8)
                ax.annotate(name, (lon, lat),
                            textcoords='offset points', xytext=(4, 4),
                            fontsize=7, fontweight='bold',
                            color=color, zorder=6)

        ax.set_xlim(plot_extent[0], plot_extent[1])
        ax.set_ylim(plot_extent[2], plot_extent[3])
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude',  fontsize=11)
        ax.set_title(f'{dtype_label} — {comp_label} | '
                     f't = {self.time[it]:.3f} s',
                     fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'frame_{i:04d}.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Frame {i+1}/{n_frames}  (t={self.time[it]:.3f}s)')

    #  assemble video 
    try:
        ffmpeg_exe = ffmpeg_path or _shutil.which('ffmpeg') or 'ffmpeg'
        cmd = [
            ffmpeg_exe, '-y',
            '-framerate', str(fps),
            '-i', os.path.join(output_dir, 'frame_%04d.png'),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_video,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f'Video saved: {output_video}')
        else:
            print(f'ffmpeg failed (exit {result.returncode})')
            print(result.stderr[-400:])
    except Exception as e:
        print(f'ffmpeg error: {e}')
