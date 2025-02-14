import os
import json
from verify import NuScenesEvaluator
import numpy as np
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory

def load_split_tokens(split_file):
    """Load sample tokens from a split file (one token per line)."""
    with open(split_file, 'r') as f:
        tokens = [line.strip() for line in f if line.strip()]
    return set(tokens)

def evaluate():
    # Path to the mini_val split file
    split_file = 'D:\\nuscenes\\v1.0-mini\\split\\mini_val.txt'
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # Load split tokens
    eval_tokens = load_split_tokens(split_file)
    print(f"Loaded {len(eval_tokens)} tokens for evaluation split.")

    # Initialize evaluator
    evaluator = NuScenesEvaluator(model_path='hybrid_detector.pth')

    # Get ground truth tokens from NuScenes
    gt_tokens = set()
    for scene in evaluator.nusc.scene:
        sample_token = scene['first_sample_token']
        while sample_token:
            gt_tokens.add(sample_token)
            sample = evaluator.nusc.get('sample', sample_token)
            sample_token = sample['next']

    # Intersection of evaluation tokens and ground truth tokens
    valid_tokens = eval_tokens.intersection(gt_tokens)
    print(f"Filtered to {len(valid_tokens)} valid tokens for evaluation.")

    # Prepare results dictionary
    results_dict = {}

    # Evaluate predictions for the specified split
    for sample_token in valid_tokens:
        sample_idx = evaluator.dataset.get_sample_index(sample_token)
        if sample_idx is None:
            print(f"Sample token {sample_token} not found in dataset.")
            continue

        # Run inference
        pred_boxes, _ = evaluator.predict(sample_idx)
        results_dict[sample_token] = []

        # Format predictions for NuScenes evaluation
        for box in pred_boxes:
            raw_class_name = list(evaluator.dataset.class_map.keys())[box['class_id']]
            mapped_name = evaluator.detection_name_mapping.get(raw_class_name, raw_class_name)

            detection = {
                'sample_token': sample_token,
                'translation': [float(box['x']), float(box['y']), float(box['z'])],
                'size': [float(box['l']), float(box['w']), float(box['height'])],
                'rotation': [float(np.cos(box['yaw'] / 2.0)), 0, 0, float(np.sin(box['yaw'] / 2.0))],
                'detection_name': mapped_name,
                'detection_score': float(box['confidence']),
                'velocity': [0.0, 0.0],  # Default velocity
                'attribute_name': ""     # Default attribute
            }
            results_dict[sample_token].append(detection)

    # Meta information required by NuScenes
    meta = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False
    }

    # Final results dictionary
    final_results = {"results": results_dict, "meta": meta}

    # Save results to a JSON file
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f)
    print(f"Results saved to {output_file}.")

    # Verify that predictions match ground truth tokens
    assert set(results_dict.keys()) == gt_tokens, "Mismatch between prediction and ground truth tokens"

    # Run official NuScenes evaluation
    nusc_eval = NuScenesEval(
        evaluator.nusc,
        config=config_factory('detection_cvpr_2019'),
        result_path=output_file,
        eval_set='mini_val',
        output_dir='./eval_results'
    )
    metrics = nusc_eval.main()
    print("Evaluation Metrics:")
    print(metrics)