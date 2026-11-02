"""Post-v6 RIGHT adapter for the opt-in75deg stand candidate, no v6 edits."""
from t1_right_station_yaw_candidate_v7 import prepare_standing_right_v7_yaw
import t1_right_standing_motion_v6 as original

def _once(source,old,new):
    if source.count(old)!=1:raise RuntimeError('V7 yaw75 strict source anchor mismatch: '+repr(old)+' count='+str(source.count(old)))
    return source.replace(old,new,1)

def make_standing_preparer_v7_yaw(config):
    if config.stand_yaw_deg!=75. or config.aspiration_tool_world_yaw_delta_deg!=15.:raise ValueError('V7 standalone-tested candidate requires stand75/turn15')
    def prepare(m,d,out,plan,*,time_scale=1.25):return prepare_standing_right_v7_yaw(m,d,out,plan,time_scale=time_scale,config=config)
    return prepare

def adapt_neutral_source_v7_yaw(source,config):
    if config.stand_yaw_deg!=75. or config.aspiration_tool_world_yaw_delta_deg!=15.:raise ValueError('Frozen layout must explicitly contain stand75 and turn15')
    source=original.adapt_neutral_source_v6(source,config)
    return _once(source,'from t1_right_standing_motion_v6 import make_standing_preparer_v6','from t1_right_station_yaw_launch_v7 import make_standing_preparer_v7_yaw as make_standing_preparer_v6')

def adapt_launch_source(source):
    """Call AFTER existing v6 RIGHT adapter and BEFORE root layout-last adapter.

    Root coherent layout must compile stand_yaw_deg75/cfg.stand75/aspdelta15;
    this changes only the hidden preparer yaw anchor through a private copy.
    """
    return _once(source,'from t1_right_standing_motion_v6 import adapt_neutral_source_v6','from t1_right_station_yaw_launch_v7 import adapt_neutral_source_v7_yaw as adapt_neutral_source_v6')
