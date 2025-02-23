# import base64
from flask import Flask, request, jsonify, send_file
import torch
import requests
import clip
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
import options.option_transformer as option_trans
import visualization.plot_3d_global as plot_3d
import os

from utils.motion_process import recover_from_ric

app = Flask(__name__)

# Global variables to store models
global_models = {}

def initialize_models():
    """Initialize all required models and store them in global_models dictionary"""
    args = option_trans.get_args_parser()
    args.dataname = 't2m'
    args.resume_pth = 'pretrained/vq_best.pth'
    args.resume_trans = 'pretrained/net_best_fid.pth'
    args.down_t = 2
    args.depth = 3
    args.block_size = 51

    # Initialize CLIP
    clip_model, _ = clip.load("ViT-B/32", device=torch.device('cuda'), jit=True, download_root='./')
    # clip_model, _ = build_model(r"C:\Users\advay\Hackalytics\VQ-Trans\pretrained\clip_best.pth", device=torch.device('cuda'), jit=True)
    clip_model.load_state_dict(torch.load("pretrained/clip_best.pth", map_location='cuda'), strict=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Initialize VQVAE
    net = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate
    )

    # Initialize Transformer
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code,
        embed_dim=1024,
        clip_dim=args.clip_dim,
        block_size=args.block_size,
        num_layers=9,
        n_head=16,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.ff_rate
    )

    # Load checkpoints
    ckpt = torch.load(args.resume_pth, map_location='cuda')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    ckpt = torch.load(args.resume_trans, map_location='cuda')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval()
    trans_encoder.cuda()

    # Load mean and std
    mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

    # Store all models and parameters in global dictionary
    global_models.update({
        'clip_model': clip_model,
        'net': net,
        'trans_encoder': trans_encoder,
        'mean': mean,
        'std': std
    })

@app.route('/generate_motion', methods=['POST'])
def generate_motion():
    try:
        # Get text input from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        clip_text = [data['text']]

        # Generate motion
        with torch.no_grad():
            # Encode text using CLIP
            text = clip.tokenize(clip_text, truncate=True).cuda()
            feat_clip_text = global_models['clip_model'].encode_text(text).float()

            # Generate motion indices
            index_motion = global_models['trans_encoder'].sample(feat_clip_text[0:1], False)
            pred_pose = global_models['net'].forward_decoder(index_motion)

            # Convert to xyz coordinates
            pred_xyz = recover_from_ric(
                (pred_pose * global_models['std'] + global_models['mean']).float(),
                22
            )
            xyz = pred_xyz.reshape(1, -1, 22, 3)

            # Save motion data and generate visualization
            output_dir = 'generated'
            os.makedirs(output_dir, exist_ok=True)
            
            # Use unique filename based on timestamp or request ID
            import time
            timestamp = int(time.time() * 1000)
            motion_path = os.path.join(output_dir, f'motion_{timestamp}.npy')
            gif_path = os.path.join(output_dir, f'animation_{timestamp}.gif')
            
            # Save the motion data
            np.save(motion_path, xyz.detach().cpu().numpy())
            
            # Generate visualization
            plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), clip_text, [gif_path])

            # Return the path that can be used to fetch the GIF
            return jsonify({
                'message': 'Motion generated successfully',
                'animation_id': timestamp
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_animation/<int:animation_id>', methods=['GET'])
def get_animation(animation_id):
    """Endpoint to retrieve the generated animation"""
    try:
        gif_path = os.path.join('generated', f'animation_{animation_id}.gif')
        if not os.path.exists(gif_path):
            return jsonify({'error': 'Animation not found'}), 404
            
        return send_file(
            gif_path,
            mimetype='image/gif',
            as_attachment=False,
            download_name=f'animation_{animation_id}.gif'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_motion/<int:animation_id>', methods=['GET'])
def get_motion(animation_id):
    """Endpoint to retrieve the motion data"""
    try:
        motion_path = os.path.join('generated', f'motion_{animation_id}.npy')
        if not os.path.exists(motion_path):
            return jsonify({'error': 'Motion data not found'}), 404
            
        return send_file(
            motion_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'motion_{animation_id}.npy'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
# Optional: Add cleanup routine to prevent disk space issues
def cleanup_old_files(max_age_hours=24):
    """Delete files older than max_age_hours"""
    import time
    current_time = time.time()
    
    for filename in os.listdir('generated'):
        filepath = os.path.join('generated', filename)
        if os.path.getmtime(filepath) < current_time - (max_age_hours * 3600):
            try:
                os.remove(filepath)
            except Exception:
                pass

if __name__ == '__main__':
    # Initialize models before starting the server
    initialize_models()
    # Create generated directory if it doesn't exist
    os.makedirs('generated', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)