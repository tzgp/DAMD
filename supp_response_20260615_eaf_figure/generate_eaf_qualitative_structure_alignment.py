# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from pathlib import Path
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_defaults import default_dataset_root

from dataset import get_data_loader
from experiments.revision.run_eaf_analysis import AnalysisLateFusion, _overlay_map_on_rgb, _minmax, build_parser
from utils.mvtec3d_util import read_tiff_organized_pc, organized_pc_to_depth_map

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / 'figures'
ART_DIR = BASE_DIR / 'artifacts'
FIG_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        'category': 'carrot',
        'defect_type': 'combined',
        'sample_id': '010',
    },
    {
        'category': 'bagel',
        'defect_type': 'contamination',
        'sample_id': '008',
    },
    {
        'category': 'rope',
        'defect_type': 'open',
        'sample_id': '000',
    },
]


def _apply_style():
    serif_stack = ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif', 'STIXGeneral']
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': serif_stack,
        'mathtext.fontset': 'stix',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#d9d9d9',
        'text.color': '#1f1f1f',
    })
    resolved = font_manager.FontProperties(family='serif').get_name()
    print(f'[qualitative] serif font resolved to: {resolved}')



def _build_args():
    parser = build_parser()
    args = parser.parse_args([])
    args.dataset = 'mvtec3d'
    args.dataset_type = 'mvtec3d'
    args.data_root = str(default_dataset_root('mvtec3d'))
    args.dataset_path = str(default_dataset_root('mvtec3d'))
    args.method_name = 'DINO+FPFH+add+late'
    args.save_feature = False
    args.use_uff = False
    args.use_trans = True
    args.use_ssl = False
    args.save_preds = True
    args.include_ocsvm = True
    return args


def _collect_selected_samples(args):
    selected_categories = sorted({case['category'] for case in CASES})
    selected_lookup = {(case['category'], case['defect_type'], case['sample_id']): case for case in CASES}
    found = {}
    for category in selected_categories:
        print(f'[qualitative] processing category: {category}')
        method = AnalysisLateFusion(args)
        train_loader = get_data_loader('train', class_name=category, img_size=args.img_size, args=args)
        for sample, _ in train_loader:
            method.add_sample_to_mem_bank(sample)
        method.run_coreset()
        train_loader_for_ocsvm = get_data_loader('train', class_name=category, img_size=args.img_size, args=args)
        for sample, _ in train_loader_for_ocsvm:
            method.collect_ocsvm_training_sample(sample)
        method.train_ocsvm_fusion()
        test_loader = get_data_loader('test', class_name=category, img_size=args.img_size, args=args)
        with torch.no_grad():
            for sample, mask, label, rgb_path in test_loader:
                method.set_context(category, rgb_path, label)
                method.predict(sample, mask, label)
        for artifact in method.sample_artifacts:
            key = (artifact.category, artifact.defect_type, artifact.sample_id)
            if key in selected_lookup:
                found[key] = artifact
    missing = [key for key in selected_lookup if key not in found]
    if missing:
        raise RuntimeError(f'Missing requested cases: {missing}')
    return [found[(case['category'], case['defect_type'], case['sample_id'])] for case in CASES]


def _depth_from_rgb_path(rgb_path: str, target_shape):
    tiff_path = rgb_path.replace('/rgb/', '/xyz/').replace('.png', '.tiff')
    organized_pc = read_tiff_organized_pc(tiff_path)
    depth_map = organized_pc_to_depth_map(organized_pc).astype(np.float64)
    depth_map = _minmax(depth_map)
    depth_img = Image.fromarray((depth_map * 255.0).astype(np.uint8)).resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(depth_img, dtype=np.float64) / 255.0


def _save_manifest(selected_samples):
    manifest_path = ART_DIR / 'selected_cases_manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'defect_type', 'sample_id', 'rgb_path', 'lambda_rgb', 'lambda_pc', 'score_rgb', 'score_pc', 'score_eaf'])
        for case, sample in zip(CASES, selected_samples):
            writer.writerow([
                sample.category, sample.defect_type, sample.sample_id, sample.rgb_path,
                f'{sample.lambda_rgb:.6f}', f'{sample.lambda_pc:.6f}', f'{sample.score_rgb:.6f}', f'{sample.score_pc:.6f}', f'{sample.score_eaf:.6f}'
            ])
    return manifest_path


def _make_figure(selected_samples):
    n_rows = len(selected_samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(9.2, 2.02 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample in enumerate(selected_samples):
        rgb_pil = Image.open(sample.rgb_path).convert('RGB').resize((sample.map_eaf.shape[1], sample.map_eaf.shape[0]), Image.BILINEAR)
        rgb_image = np.asarray(rgb_pil, dtype=np.float64) / 255.0
        depth_map = _depth_from_rgb_path(sample.rgb_path, sample.map_eaf.shape)
        overlay = _overlay_map_on_rgb(rgb_image, sample.map_eaf)
        images = [
            sample.gt_mask,
            sample.map_rgb,
            sample.map_pc,
            overlay,
        ]
        cmaps = ['gray', 'jet', 'jet', None]
        for col_idx, (img, cmap) in enumerate(zip(images, cmaps)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap=cmap)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor('#dddddd')

        case_line = f"{sample.category} | {sample.defect_type} | {sample.sample_id}"
        axes[row_idx, 0].text(
            0.0,
            1.01,
            case_line,
            transform=axes[row_idx, 0].transAxes,
            ha='left',
            va='bottom',
            fontsize=9.2,
            fontstyle='italic',
        )

    fig.subplots_adjust(left=0.016, right=0.999, top=0.992, bottom=0.006, wspace=0.008, hspace=0.008)
    outputs = []
    for suffix in ('png', 'pdf'):
        output_path = FIG_DIR / f'eaf_qualitative_structure_alignment.{suffix}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        outputs.append(output_path)
        print(output_path)
    plt.close(fig)
    return outputs


def main():
    _apply_style()
    args = _build_args()
    selected_samples = _collect_selected_samples(args)
    _save_manifest(selected_samples)
    _make_figure(selected_samples)


if __name__ == '__main__':
    main()
