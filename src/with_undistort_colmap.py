#!/usr/bin/env python3

import subprocess
import time
from datetime import datetime
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval

def save_timing_info(outputs, timing_info, model_stats):
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_dir = outputs / 'stats'
    stats_dir.mkdir(exist_ok=True)
    
    stats_file = stats_dir / f'timing_stats_{timestamp}.md'
    
    with open(stats_file, 'w') as f:
        f.write(f"# Pipeline Statistics - {timestamp}\n\n")
        f.write("## Timing Information\n\n")
        f.write("| Step | Time (seconds) |\n")
        f.write("|------|----------------|\n")
        
        total_time = 0
        for step, duration in timing_info.items():
            f.write(f"| {step} | {duration:.2f} |\n")
            if step != "Total":
                total_time += duration
                
        f.write(f"| **Total** | **{total_time:.2f}** |\n\n")
        
        f.write("## COLMAP Model Statistics\n\n")
        f.write("```\n")
        f.write(model_stats)
        f.write("\n```\n")

def main():
    timing_info = {}
    start_total = time.time()
    
    base = Path("/home/somusan/dev-somusan/classical_cv/3d_vision/3dgs/dataset/scannet_imp/dataset/4a1a3a7dc5_org/4a1a3a7dc5/fps_extracted/undistortion_for_high_Res/")
    images = base / Path('images/')
    outputs = base / Path('./HLOC_2kALIKED+lightglue_Vt8')  # Incremented version
    outputs.mkdir(exist_ok=True)

    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    sfm_dir.mkdir(exist_ok=True)
    
    features = outputs / 'feats-aliked-n16.h5'
    matches = outputs / 'matches-aliked-lightglue.h5'

    try:
        # Time NetVLAD feature extraction
        start = time.time()
        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        timing_info["NetVLAD Extraction"] = time.time() - start

        # Time pair generation
        start = time.time()
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=50)
        timing_info["Pair Generation"] = time.time() - start

        # Time ALIKED feature extraction
        start = time.time()
        feature_conf = extract_features.confs['aliked-n16']
        feature_conf["preprocessing"]["resize_max"] = 512
        extract_features.main(feature_conf, images, outputs)
        timing_info["ALIKED Feature Extraction"] = time.time() - start

        # Time feature matching
        start = time.time()
        matcher_conf = match_features.confs['aliked+lightglue']
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        timing_info["Feature Matching"] = time.time() - start

        references = sorted([
            str(p) for p in images.glob('*')
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        # Time reconstruction
        start = time.time()
        reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
        timing_info["COLMAP Reconstruction"] = time.time() - start

        # Time undistortion
        start = time.time()
        undistorted_dir = outputs / 'undistorted'
        undistorted_dir.mkdir(exist_ok=True)

        colmap_cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(images),
            '--output_path', str(undistorted_dir),
            '--input_path', str(sfm_dir),
            '--max_image_size', '1024'
        ]
        subprocess.run(colmap_cmd, check=True)
        timing_info["Image Undistortion"] = time.time() - start

        # Run COLMAP model_analyzer
        model_stats = ""
        try:
            analyzer_cmd = [
                'colmap', 'model_analyzer',
                '--path', str(sfm_dir / 'sparse/0')
            ]
            model_stats = subprocess.check_output(analyzer_cmd, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            model_stats = f"Error running model_analyzer: {str(e)}"

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        model_stats = f"Error during processing: {str(e)}"
    finally:
        timing_info["Total"] = time.time() - start_total
        
        # Save timing and model statistics
        save_timing_info(outputs, timing_info, model_stats)
        
        # Print timing information
        print("\nTiming Information:")
        print("-" * 40)
        for step, duration in timing_info.items():
            print(f"{step}: {duration:.2f} seconds")

if __name__ == '__main__':
    main()
