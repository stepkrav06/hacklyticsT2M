import torch 
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    
    joints, out_name, title = args
    
    data = joints.copy().reshape(len(joints), -1, 3)
    
    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    
    colors = ['#4B0082',  # Indigo
             '#5D3FD3',   # Dark indigo
             '#E6E6FA',   # Light indigo
             '#4B0082',   # Indigo
             '#5D3FD3',   # Dark indigo
             '#9683EC',   # Medium indigo
             '#9683EC',   # Medium indigo
             '#9683EC',   # Medium indigo
             '#9683EC',   # Medium indigo
             '#9683EC',   # Medium indigo
             '#7A5DC7',   # Deep indigo
             '#7A5DC7',   # Deep indigo
             '#7A5DC7',   # Deep indigo
             '#7A5DC7',   # Deep indigo
             '#7A5DC7']   # Deep indigo
    
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)
            
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            
            ax.xaxis.set_pane_color((0, 0, 0, 1.0))
            ax.yaxis.set_pane_color((0, 0, 0, 1.0))
            ax.zaxis.set_pane_color((0, 0, 0, 1.0))

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            # Increased opacity to 0.5 and lightened the indigo color
            xz_plane.set_facecolor((0.4, 0.3, 0.6, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10), dpi=96)
        ax = p3.Axes3D(fig)
        
        init()
        
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='#9683EC')  # Medium indigo for trajectory

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
                     linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
        if out_name is not None:
            plt.savefig(out_name, dpi=96, facecolor='black', edgecolor='none')
            plt.close()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96, facecolor='black', edgecolor='none')
            io_buf.seek(0)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                           newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)

def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None):
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out