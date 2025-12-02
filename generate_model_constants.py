#!/usr/bin/env python3
"""
Generate HLS-compatible C++ constants from trained MiniRocket model
"""
import json
import sys

def generate_hls_constants(model_file, output_file):
    """Convert JSON model to HLS C++ constants"""
    
    with open(model_file, 'r') as f:
        model = json.load(f)
    
    with open(output_file, 'w') as f:
        f.write('#ifndef MINIROCKET_MODEL_CONSTANTS_H\n')
        f.write('#define MINIROCKET_MODEL_CONSTANTS_H\n\n')
        f.write('#include "minirocket_inference_hls.h"\n\n')
        f.write('// Auto-generated model constants from training\n\n')
        
        # Model dimensions
        f.write(f'const int_t MODEL_NUM_FEATURES = {model["num_features"]};\n')
        f.write(f'const int_t MODEL_NUM_CLASSES = {model["num_classes"]};\n')
        f.write(f'const int_t MODEL_NUM_DILATIONS = {model["num_dilations"]};\n\n')
        
        # Dilations array
        f.write('const int_t MODEL_DILATIONS[MAX_DILATIONS] = {\n    ')
        dilations = model["dilations"]
        for i, d in enumerate(dilations):
            f.write(f'{d}')
            if i < len(dilations) - 1:
                f.write(', ')
        f.write('\n};\n\n')
        
        # Features per dilation
        f.write('const int_t MODEL_FEATURES_PER_DILATION[MAX_DILATIONS] = {\n    ')
        features_per_dil = model["num_features_per_dilation"]
        for i, f_count in enumerate(features_per_dil):
            f.write(f'{f_count}')
            if i < len(features_per_dil) - 1:
                f.write(', ')
        f.write('\n};\n\n')
        
        # Biases (truncated for large arrays)
        biases = model["biases"]
        f.write(f'const data_t MODEL_BIASES[{len(biases)}] = {{\n')
        for i in range(0, len(biases), 8):  # 8 per line
            f.write('    ')
            for j in range(min(8, len(biases) - i)):
                f.write(f'{biases[i+j]:.6f}f')
                if i + j < len(biases) - 1:
                    f.write(', ')
            f.write('\n')
        f.write('};\n\n')
        
        # Scaler parameters
        scaler_mean = model["scaler_mean"]
        f.write(f'const data_t MODEL_SCALER_MEAN[{len(scaler_mean)}] = {{\n')
        for i in range(0, len(scaler_mean), 8):
            f.write('    ')
            for j in range(min(8, len(scaler_mean) - i)):
                f.write(f'{scaler_mean[i+j]:.6f}f')
                if i + j < len(scaler_mean) - 1:
                    f.write(', ')
            f.write('\n')
        f.write('};\n\n')
        
        scaler_scale = model["scaler_scale"]
        f.write(f'const data_t MODEL_SCALER_SCALE[{len(scaler_scale)}] = {{\n')
        for i in range(0, len(scaler_scale), 8):
            f.write('    ')
            for j in range(min(8, len(scaler_scale) - i)):
                f.write(f'{scaler_scale[i+j]:.6f}f')
                if i + j < len(scaler_scale) - 1:
                    f.write(', ')
            f.write('\n')
        f.write('};\n\n')
        
        # Classifier intercept
        intercept = model["classifier_intercept"]
        f.write('const data_t MODEL_INTERCEPT[MAX_CLASSES] = {\n    ')
        for i, val in enumerate(intercept):
            f.write(f'{val:.6f}f')
            if i < len(intercept) - 1:
                f.write(', ')
        f.write('\n};\n\n')
        
        # Classifier coefficients (this is the big one!)
        coef = model["classifier_coef"]
        f.write(f'// WARNING: Large coefficient array - {len(coef)} x {len(coef[0])} elements\n')
        f.write(f'// Consider using external memory or tiling for large models\n')
        f.write(f'const data_t MODEL_COEFFICIENTS[{len(coef)}][{len(coef[0])}] = {{\n')
        
        for class_idx, class_coef in enumerate(coef):
            f.write(f'    {{ // Class {class_idx}\n')
            for i in range(0, len(class_coef), 8):
                f.write('        ')
                for j in range(min(8, len(class_coef) - i)):
                    f.write(f'{class_coef[i+j]:.6f}f')
                    if i + j < len(class_coef) - 1:
                        f.write(', ')
                f.write('\n')
            f.write('    }')
            if class_idx < len(coef) - 1:
                f.write(',')
            f.write('\n')
        
        f.write('};\n\n')
        f.write('#endif // MINIROCKET_MODEL_CONSTANTS_H\n')
    
    print(f"Generated HLS constants in {output_file}")
    print(f"Model size: {len(coef)} classes × {len(coef[0])} features = {len(coef) * len(coef[0])} coefficients")
    print(f"Estimated BRAM usage: ~{(len(coef) * len(coef[0]) * 4) // 1024} KB")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python generate_model_constants.py model.json output.h")
        sys.exit(1)
    
    generate_hls_constants(sys.argv[1], sys.argv[2])