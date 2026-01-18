#!/usr/bin/env python3
"""
Synthetic ToF Data Generation Script
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_tof import SyntheticToFGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic ToF Data')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./data/train',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--width', type=int, default=640,
                        help='Image width')
    parser.add_argument('--height', type=int, default=480,
                        help='Image height')
    
    args = parser.parse_args()
    
    print(f'Generating {args.num_samples} synthetic ToF samples...')
    print(f'Resolution: {args.width}x{args.height}')
    print(f'Output: {args.output_dir}')
    
    generator = SyntheticToFGenerator(
        width=args.width,
        height=args.height,
        seed=args.seed
    )
    
    generator.generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        balanced=True
    )
    
    print('Done!')


if __name__ == '__main__':
    main()
