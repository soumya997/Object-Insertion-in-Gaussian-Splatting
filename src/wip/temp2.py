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
    outputs = base / Path('./HLOC_2kALIKED+lightglue_V9')  # Incremented version
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
        pairs_from_retrieval.main(
            retrieval_path, 
            sfm_pairs, 
            num_matched=25  # Reduced from 50 to reduce complexity
        )
        timing_info["Pair Generation"] = time.time() - start

        # Time ALIKED feature extraction
        start = time.time()
        feature_conf = extract_features.confs['aliked-n16']
        feature_conf["preprocessing"]["resize_max"] = 512
        extract_features.main(feature_conf, images, outputs)
        timing_info["ALIKED Feature Extraction"] = time.time() - start

        # Time feature matching with modified parameters
        start = time.time()
        matcher_conf = match_features.confs['aliked+lightglue']
        matcher_conf.update({
            'max_error': 4.0,
            'confidence': 0.999,
            'min_inlier_ratio': 0.15,
            'min_num_inliers': 15,
            'max_num_trials': 10000,
            'max_epipolar_error': 4.0,
        })
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        timing_info["Feature Matching"] = time.time() - start

        references = sorted([
            str(p) for p in images.glob('*')
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        # Time reconstruction with single camera model
        start = time.time()
        try:
            # First attempt: Using HLOC's reconstruction with basic options
            reconstruction.main(
                sfm_dir, 
                images, 
                sfm_pairs, 
                features, 
                matches, 
                image_list=references,
                mapper_options={
                    'num_threads': 16,
                    'multiple_models': 0,
                    'min_model_size': 3,
                }
            )
        except Exception as e:
            print(f"HLOC reconstruction failed: {str(e)}")
            print("Falling back to direct COLMAP mapper...")
            
            # Create COLMAP database
            db_path = sfm_dir / "database.db"
            
            # Run COLMAP feature_extractor
            feature_extractor_cmd = [
                'colmap', 'feature_extractor',
                '--database_path', str(db_path),
                '--image_path', str(images),
                '--ImageReader.single_camera', '1',
                '--ImageReader.camera_model', 'SIMPLE_RADIAL'
            ]
            subprocess.run(feature_extractor_cmd, check=True)
            
            # Run COLMAP exhaustive_matcher
            matcher_cmd = [
                'colmap', 'exhaustive_matcher',
                '--database_path', str(db_path),
            ]
            subprocess.run(matcher_cmd, check=True)
            
            # Run COLMAP mapper with single camera
            mapper_cmd = [
                'colmap', 'mapper',
                '--database_path', str(db_path),
                '--image_path', str(images),
                '--output_path', str(sfm_dir),
                '--Mapper.single_camera', '1',
                '--Mapper.init_min_tri_angle', '4.0',
                '--Mapper.ba_global_images_ratio', '1.1',
                '--Mapper.ba_global_points_ratio', '1.1',
                '--Mapper.filter_max_reproj_error', '4.0',
                '--Mapper.min_num_matches', '15',
                '--Mapper.abs_pose_min_num_inliers', '15',
                '--Mapper.abs_pose_min_inlier_ratio', '0.15',
                '--Mapper.abs_pose_max_error', '12.0',
                '--Mapper.filter_min_tri_angle', '1.5'
            ]
            subprocess.run(mapper_cmd, check=True)
            
        timing_info["COLMAP Reconstruction"] = time.time() - start

        # Time undistortion
        start = time.time()
        undistorted_dir = outputs / 'undistorted'
        undistorted_dir.mkdir(exist_ok=True)

        # Modified COLMAP command with single camera
        colmap_cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(images),
            '--output_path', str(undistorted_dir),
            '--input_path', str(sfm_dir),
            '--max_image_size', '1024',
            '--single_camera', '1'
        ]
        subprocess.run(colmap_cmd, check=True)

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
