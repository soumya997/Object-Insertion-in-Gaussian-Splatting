#!/usr/bin/env python3

import subprocess
from pathlib import Path

from hloc import extract_features, match_features, pairs_from_retrieval

def main():
    base = Path("/home/somusan/dev-somusan/classical_cv/3d_vision/3dgs/dataset/scannet_imp/dataset/4a1a3a7dc5_org/4a1a3a7dc5/fps_extracted/undistortion_for_high_Res/")
    images = base / Path('images/')
    outputs = base / Path('./HLOC_2kALIKED+lightglue_vt2')  # Incremented version
    outputs.mkdir(exist_ok=True, parents=True)

    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    sfm_dir.mkdir(exist_ok=True)
    
    # Clean up and create feature/match paths
    feature_path = outputs / 'feats-aliked-n16.h5'
    matches_path = outputs / 'matches-aliked-lightglue.h5'
    
    # Remove existing files if they exist
    for path in [feature_path, matches_path]:
        if path.exists():
            import shutil
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    
    print(f"Features will be saved to: {feature_path}")
    print(f"Matches will be saved to: {matches_path}")

    # Extract NetVLAD features for image retrieval
    retrieval_conf = extract_features.confs["netvlad"]
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)

    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=25)  # Reduced number of pairs

    # Extract ALIKED features for matching
    feature_conf = extract_features.confs['aliked-n16']
    feature_conf["preprocessing"]["resize_max"] = 320
    print(f"Feature extraction configuration: {feature_conf}")
    extract_features.main(feature_conf, images, outputs)
    
    # Verify feature file
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file was not created at {feature_path}")
    print(f"Feature file created successfully, size: {feature_path.stat().st_size} bytes")

    # Match features with modified parameters
    matcher_conf = match_features.confs['aliked+lightglue']
    matcher_conf.update({
        'max_error': 4.0,
        'confidence': 0.999,
        'min_inlier_ratio': 0.15,
        'min_num_inliers': 15,
    })
    match_features.main(matcher_conf, sfm_pairs, features=feature_path, matches=matches_path)

    # Create COLMAP database
    db_path = sfm_dir / "database.db"
    if db_path.exists():
        db_path.unlink()
    
    # Run COLMAP feature_extractor with single camera
    feature_extractor_cmd = [
        'colmap', 'feature_extractor',
        '--database_path', str(db_path),
        '--image_path', str(images),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'SIMPLE_RADIAL',
        '--ImageReader.camera_params', '2048,1024,512'  # Default focal length and principal point
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
        '--Mapper.single_camera', '1'
    ]
    subprocess.run(mapper_cmd, check=True)

    # Run undistortion
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

if __name__ == '__main__':
    main()