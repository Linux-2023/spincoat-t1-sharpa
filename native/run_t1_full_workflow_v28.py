"""Fresh V10-based full workflow with V27 lateral spatial shaking.
Acceptance and original liquid proxy retained; never splices episodes.
"""
import importlib.util
import json
import sys
from pathlib import Path
from t1_v10_v27_stage_v28 import adapt_launch
ROOT=Path('/home/agilex/spincoat-t1-sharpa')

def main():
    import t1_single_grasp_v10 as grasp
    baseline_path=ROOT/'research/hardware/releases/v10_09_accepted_grasp/run_t1_human_workflow_v10.py'
    spec=importlib.util.spec_from_file_location('accepted_v10_full_baseline',baseline_path)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    original=grasp.adapt_launch
    def combined(source):return adapt_launch(original(source))
    grasp.adapt_launch=combined
    tag=sys.argv[sys.argv.index('--tag')+1];out=ROOT/'logs/spincoat/t1'/tag
    try:baseline.main()
    finally:
        grasp.adapt_launch=original
        if out.exists():
            frozen=ROOT/'research/hardware/human_workflow_v7/frozen_inputs'/tag
            overlay_applied=(frozen/'grip_overlay.json').exists()
            (out/'v28_full_workflow_provenance.json').write_text(json.dumps(dict(scope=__doc__,baseline='v10_09_accepted_grasp',motion_module='t1_wrist_mixing_v27',motion_frequency_hz=3.,motion_amplitude_deg=5.,v10_grip_and_layout_retained=not overlay_applied,grip_overlay_applied=overlay_applied,grip_overlay_path=str(frozen/'grip_overlay.json') if overlay_applied else None,v27_upper_grip_overlay_applied=False,v15_preload_applied=False,later_wrist_reset_profiles_applied=False,liquid_model='original V10 pose-gated conservative volume proxy, not V27 modal liquid',planned_mixing_stop=False,original_acceptance_retained=True,no_cross_episode_splicing=True,full_success_not_inferred=True),indent=2))
            if (frozen/'hold_profile.json').exists():
                (out/'v30_hold_provenance.json').write_text(json.dumps(dict(baseline='V10_09',motion='V27',hold_profile=json.loads((frozen/'hold_profile.json').read_text()),grip_overlay_applied=overlay_applied,model_parameters_modified=False,scope='Bounded actuator commands only; success requires actual post-shake transport and pour verification'),indent=2))

if __name__=='__main__':main()
