"""
animation_plots.py
==================
Standalone matplotlib animation helpers for ``ShakerMakerData``.

Both helpers render PNG frames to disk and then call ``ffmpeg`` to stitch
them into an MP4. ``ffmpeg`` is located via the ``ffmpeg_path`` parameter
or, when not given, via ``shutil.which('ffmpeg')`` on ``$PATH``.

Heavy by design: a single 200-frame animation can write 200 PNGs to
``output_dir``. The directory is created if missing and existing frames
are silently overwritten.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ...utils import _rotate

def create_animation(self, time_start=0.0, time_end=None, n_frames=50,
                     component='z', data_type='vel', cmap='RdBu_r',
                     figsize=(12,8), dpi=100, fps=10,
                     elev=30, azim=-60, s=20, alpha=0.85,
                     ffmpeg_path=None, output_dir='animation', output_video='animation.mp4',
                     axis_equal=True, vmax_from_range=False):

    """Render a 3-D scatter animation of the full domain to an MP4.

    Parameters
    ----------
    self : ShakerMakerData
        Bound when this function is exposed as a method.
    time_start : float, default ``0.0``
        First simulation time included in the animation.
    time_end : float, optional
        Last simulation time. Defaults to ``self.time[-1]``.
    n_frames : int, default ``50``
        Number of frames sampled uniformly between ``time_start`` and
        ``time_end``.
    component : {'z', 'e', 'n', 'resultant'}, default ``'z'``
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    cmap : str, default ``'RdBu_r'``
    figsize, dpi, fps : matplotlib / ffmpeg settings.
    elev, azim : float
        Initial 3-D view angles.
    s, alpha : float
        Scatter marker size and opacity.
    ffmpeg_path : str, optional
        Full path to the ``ffmpeg`` binary. Defaults to the system one.
    output_dir : str, default ``'animation'``
        Directory where frame PNGs are written.
    output_video : str, default ``'animation.mp4'``
        Final video path.
    axis_equal : bool, default ``True``
    vmax_from_range : bool, default ``False``
        When ``True`` the colour limits are computed over the requested
        ``[time_start, time_end]`` window (chunked HDF5 scan); when
        ``False`` they come from the cached global ``self._vmax``.

    Returns
    -------
    None
        The MP4 is written to ``output_video``. On ffmpeg failure the
        PNG frames are left on disk and a message is printed.
    """
    # Ensure vmax is computed
    if self._vmax is None:
        self._compute_vmax()
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    if time_end is None: time_end = self.time[-1]

    if vmax_from_range:
        i0   = int(np.argmin(np.abs(self.time - time_start)))
        i1   = int(np.argmin(np.abs(self.time - time_end)))
        path = {'accel': f'{self._data_grp}/acceleration',
                'vel':   f'{self._data_grp}/velocity',
                'disp':  f'{self._data_grp}/displacement'}[data_type]
        _chunk_rows = 600
        vmax = 0.0
        with h5py.File(self.filename, 'r') as f:
            n_rows = f[path].shape[0]
            for _s in range(0, n_rows, _chunk_rows):
                _e  = min(_s + _chunk_rows, n_rows)
                _d  = f[path][_s:_e, i0:i1+1]
                if component.lower() == 'resultant':
                    _ed = _d[0::3,:]; _nd = _d[1::3,:]; _zd = _d[2::3,:]
                    vmax = max(vmax,
                               float(np.sqrt(_ed**2+_nd**2+_zd**2).max()))
                else:
                    _row = {'e': 0, 'n': 1, 'z': 2}[component.lower()]
                    vmax = max(vmax, float(np.abs(_d[_row::3,:]).max()))
        vmin = 0 if component.lower() == 'resultant' else -vmax
    else:
        if component.lower() == 'resultant':
            vmax = self._vmax[data_type]['resultant']; vmin = 0
        else:
            vmax = self._vmax[data_type][component.lower()]; vmin = -vmax

    xyz_t = _rotate(self.xyz)
    x=xyz_t[:,0]; y=xyz_t[:,1]; z=xyz_t[:,2]

    for i,t in enumerate(np.linspace(time_start, time_end, n_frames)):
        it = int(np.argmin(np.abs(self.time - t)))
        if component.lower() == 'resultant':
            mag = np.sqrt(self.get_surface_snapshot(it,'e',data_type)**2+
                          self.get_surface_snapshot(it,'n',data_type)**2+
                          self.get_surface_snapshot(it,'z',data_type)**2)
        else:
            mag = self.get_surface_snapshot(it, component, data_type)
        fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='lightgray', s=s, alpha=0.05)
        active = np.abs(mag) >= vmax * 0.01
        if active.any():
            ax.scatter(x[active], y[active], z[active], c=mag[active],
                       cmap=cmap, s=s, alpha=alpha, vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([]); fig.colorbar(sm, ax=ax, shrink=0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f't = {self.time[it]:.3f} s', fontsize=14, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        if axis_equal:
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/frame_{i:03d}.png', dpi=dpi)
        plt.close()
        print(f'Frame {i+1}/{n_frames}')
    try:
        ffmpeg_exe = ffmpeg_path or shutil.which('ffmpeg') or 'ffmpeg'
        subprocess.run([ffmpeg_exe, '-y', '-framerate', str(fps),
                        '-i', f'{output_dir}/frame_%03d.png',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '18', output_video], check=True, capture_output=True)
        print(f'Video saved: {output_video}')
    except Exception as e:
        print(f'ffmpeg error — frames in {output_dir}: {e}')


def create_animation_plane(self, plane='xy', plane_value=0.0,
                            time_start=0.0, time_end=None, n_frames=50,
                            component='z', data_type='vel', cmap='RdBu_r',
                            figsize=(12,8), dpi=100, fps=10,
                            elev=30, azim=-60, s=50, alpha=0.85,
                            ffmpeg_path= None,
                            output_dir='animation_plane',
                            output_video='animation_plane.mp4',
                            vmax_from_range=False,
                            axis_equal=True):

    """Render an animation showing only nodes on a planar slice.

    The plane is axis-aligned and selected by ``plane`` + ``plane_value``;
    a per-node tolerance is derived from ``self.spacing`` so nodes
    slightly off-plane (numerical noise) still show.

    Parameters
    ----------
    self : ShakerMakerData
    plane : {'xy', 'xz', 'yz'}, default ``'xy'``
        Which axis is fixed. The plane value applies to the constrained
        axis (e.g. ``plane='xy', plane_value=0`` means ``z=0``).
    plane_value : float, default ``0.0``
        Coordinate of the constrained axis **in display metres** (already
        rotated -- same units the on-screen labels use).
    time_start, time_end, n_frames, component, data_type, cmap,
    figsize, dpi, fps, elev, azim, s, alpha, ffmpeg_path,
    output_dir, output_video, axis_equal, vmax_from_range :
        Same meaning as in :func:`create_animation`.

    Returns
    -------
    None
        Writes the MP4 + PNG frames. Prints a warning and returns early
        if no node falls on the requested plane.
    """
    # Ensure vmax is computed
    if self._vmax is None:
        self._compute_vmax()
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    if time_end is None: time_end = self.time[-1]
    x=self.xyz[:,0]*1000; y=self.xyz[:,1]*1000; z=self.xyz[:,2]*1000

    xyz_t = _rotate(self.xyz)
    x=xyz_t[:,0]; y=xyz_t[:,1]; z=xyz_t[:,2]

    tol = self.spacing[0]*0.1 if self.spacing[0]>0 else 1.0
    if plane.lower()=='xy':
        pmask = np.abs(z-plane_value)<tol; tpl = f'Z = {plane_value:.1f} m'
    elif plane.lower()=='xz':
        pmask = np.abs(y-plane_value)<tol; tpl = f'Y = {plane_value:.1f} m'
    elif plane.lower()=='yz':
        pmask = np.abs(x-plane_value)<tol; tpl = f'X = {plane_value:.1f} m'
    else:
        raise ValueError("plane must be 'xy','xz','yz'")
    if not pmask.any():
        print(f'No nodes found for {tpl}'); return
    pidx = np.where(pmask)[0]
    xp=x[pmask]; yp=y[pmask]; zp=z[pmask]
    if vmax_from_range:
        i0=int(np.argmin(np.abs(self.time-time_start)))
        i1=int(np.argmin(np.abs(self.time-time_end)))
        path={'accel':f'{self._data_grp}/acceleration',
              'vel':f'{self._data_grp}/velocity',
              'disp':f'{self._data_grp}/displacement'}[data_type]
        _chunk_rows = 600
        vmax = 0.0
        with h5py.File(self.filename,'r') as f:
            n_rows = f[path].shape[0]
            for _s in range(0, n_rows, _chunk_rows):
                _e  = min(_s + _chunk_rows, n_rows)
                _d  = f[path][_s:_e, i0:i1+1]
                _pidx_chunk = pidx[(pidx >= _s) & (pidx < _e)] - _s
                if len(_pidx_chunk) == 0:
                    continue
                if component.lower()=='resultant':
                    _ed=_d[0::3,:][_pidx_chunk]
                    _nd=_d[1::3,:][_pidx_chunk]
                    _zd=_d[2::3,:][_pidx_chunk]
                    vmax=max(vmax,
                             float(np.sqrt(_ed**2+_nd**2+_zd**2).max()))
                else:
                    _row={'e':0,'n':1,'z':2}[component.lower()]
                    vmax=max(vmax,
                             float(np.abs(_d[_row::3,:][_pidx_chunk]).max()))
        vmin = 0 if component.lower()=='resultant' else -vmax
    else:
        if component.lower()=='resultant':
            vmax=self._vmax[data_type]['resultant']; vmin=0
        else:
            vmax=self._vmax[data_type][component.lower()]; vmin=-vmax
    for i,t in enumerate(np.linspace(time_start,time_end,n_frames)):
        it = int(np.argmin(np.abs(self.time-t)))
        if component.lower()=='resultant':
            mag = np.sqrt(self.get_surface_snapshot(it,'e',data_type)[pidx]**2+
                          self.get_surface_snapshot(it,'n',data_type)[pidx]**2+
                          self.get_surface_snapshot(it,'z',data_type)[pidx]**2)
        else:
            mag = self.get_surface_snapshot(it,component,data_type)[pidx]
        fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111,projection='3d')
        ax.scatter(x,y,z,c='lightgray',s=5,alpha=0.05)
        active = np.abs(mag)>=vmax*0.01
        if active.any():
            ax.scatter(xp[active],yp[active],zp[active],c=mag[active],
                       cmap=cmap,s=s,alpha=alpha,vmin=vmin,vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=vmin,vmax=vmax))
        sm.set_array([]); fig.colorbar(sm,ax=ax,shrink=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        # ax.invert_zaxis()
        ax.set_title(f'{tpl} | t = {self.time[it]:.3f} s',fontsize=14,fontweight='bold')
        ax.view_init(elev=elev,azim=azim)
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/frame_{i:03d}.png',dpi=dpi)
        plt.close()
        print(f'Frame {i+1}/{n_frames}')
    try:
        ffmpeg_exe = ffmpeg_path or shutil.which('ffmpeg') or 'ffmpeg'
        subprocess.run([ffmpeg_exe, '-y', '-framerate', str(fps),
                        '-i', f'{output_dir}/frame_%03d.png',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '18', output_video], check=True, capture_output=True)
        print(f'Video saved: {output_video}')
    except Exception as e:
        print(f'ffmpeg error — frames in {output_dir}: {e}')
