"""Plan a pre-tilt, lower, and continuous pour using the robot-only geometry."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def plan(cb, scene, height=None, jet_speed=.33, lip_radius=.0115, x_adjust=0., y_adjust=0., jet_profile=False, fixed_y=None, free_roll=False, final_z=-.5):
    o=cb.robo.observe(cameras=[]);arm=cb.arms['left'];model=cb.robot
    base=dict(zip(cb.arm_names,o.arm_q));base.update(zip(cb.dex_names,o.dex_q));model.fk(base)
    P0,R0=model.pose(arm.palm);P0=P0.copy();R0=R0.copy();local=np.array(scene['mouth_local']);axis0=R0@np.array(scene['axis_local']);initial_mouth=P0+R0@local
    beaker=np.array(cb.to_root(scene['beaker_room']));m=model.m
    colliders=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in set(arm.links) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    desired_angle=float(np.arccos(np.clip(final_z/np.hypot(axis0[1],axis0[2]),-1,1))-np.arctan2(axis0[1],axis0[2]))
    angles=list(dict.fromkeys([min(2.35,desired_angle),min(2.05,desired_angle)]))
    heights=[height] if height is not None else [.10,.11,.12,.14,.16]
    last_reason=None
    for H in heights:
      for final_angle in angles:
        flow_angle=max(0.,float(np.arccos(np.clip(.15/np.hypot(axis0[1],axis0[2]),-1,1))-np.arctan2(axis0[1],axis0[2])))
        def aim(h,theta):
            if fixed_y is not None:return beaker+np.array([x_adjust,fixed_y,h])
            a=Rotation.from_rotvec([-max(theta,flow_angle),0,0]).apply(axis0)
            down=np.array([0.,0,-1.]);lip=down-np.dot(down,a)*a;lip=lip_radius*lip/max(np.linalg.norm(lip),1e-9)
            speed=(.12+(.33-.12)*np.clip(-a[2]/.25,0,1)) if jet_profile else jet_speed
            velocity=speed*a;flight=(velocity[2]+np.sqrt(velocity[2]**2+2*9.81*(h+lip[2])))/9.81
            offset=-lip[:2]-velocity[:2]*flight+np.array([x_adjust,y_adjust])
            return beaker+np.r_[offset,h]
        high_h=max(H,min(.22,float(initial_mouth[2]-beaker[2])));high=aim(high_h,0);low=aim(H,final_angle);offset=low[:2]-beaker[:2]
        first_angle=None
        for a in np.linspace(0,final_angle,240):
            d=Rotation.from_rotvec([-a,0,0]).apply(axis0)
            bottom_clearance=H-.15*max(d[2],0)-.0125*np.sqrt(max(0,1-d[2]**2))
            if bottom_clearance>=.025 and d[2]>.15:
                first_angle=float(a);break
        if first_angle is None:last_reason='no_dry_lowering_pose';continue
        targets=[]
        targets.append(('aim',[(0,initial_mouth+(high-initial_mouth)*t) for t in np.linspace(0,1,5)[1:]],2,5))
        targets.append(('pretilt',[(a,high) for a in np.linspace(0,first_angle,max(2,int(np.ceil(first_angle/.05)))+1)[1:]],2,10))
        targets.append(('lower',[(first_angle,aim(high_h+(H-high_h)*t,first_angle)) for t in np.linspace(0,1,max(2,int(np.ceil((high_h-H)/.005)))+1)[1:]],2,10))
        targets.append(('pour',[(a,aim(H,a)) for a in np.linspace(first_angle,final_angle,max(2,int(np.ceil((final_angle-first_angle)/.05)))+1)[1:]],3,120))
        targets=[item for item in targets if not (item[0]=='pretilt' and first_angle<.005) and not (item[0]=='lower' and abs(high_h-H)<.001)]
        previous=o.arm_cmd[7:].copy();phases=[];failed=False;minimum=10.;maximum_error=0.
        for name,points,min_frames,settle in targets:
          rows=[]
          for theta,mouth in points:
            D=Rotation.from_rotvec([-theta,0,0]).as_matrix();goal_R=D@R0;goal_P=mouth-goal_R@local;reference=previous.copy()
            def residual(q):
                base.update(zip(arm.names,q));model.fk(base);p,R=model.pose(arm.palm)
                if free_roll:
                    actual_point=p+R@local; actual_axis=R@np.array(scene['axis_local']); wanted_axis=goal_R@np.array(scene['axis_local'])
                    return np.r_[10*(actual_point-mouth),actual_axis-wanted_axis,.0001*(q-reference)]
                return np.r_[10*(p-goal_P),(R-goal_R).ravel(),.0001*(q-reference)]
            opt=least_squares(residual,np.clip(previous,arm.lo+1e-8,arm.hi-1e-8),bounds=(arm.lo+1e-9,arm.hi-1e-9),max_nfev=240)
            error=residual(opt.x);pose_dim=6 if free_roll else 12;maximum_error=max(maximum_error,float(np.linalg.norm(error[:pose_dim])))
            if np.linalg.norm(error[:3])>.012 or np.linalg.norm(error[3:pose_dim])>.012:
                last_reason=(name,'ik',theta,float(np.linalg.norm(error[:3])/10),float(np.linalg.norm(error[3:pose_dim])));failed=True;break
            frames=max(min_frames,int(np.ceil(np.max(np.abs(opt.x-previous)/arm.speed)*25)))
            for q in np.linspace(previous,opt.x,frames+1)[1:]:
                base.update(zip(arm.names,q));model.fk(base);z,link=arm.lowest();minimum=min(minimum,float(z))
                if z<cb.table_z+.025:last_reason=(name,'table',link,float(z));failed=True;break
                for g in colliders:
                    c=model.d.geom_xpos[g];radius=float(m.geom_rbound[g])
                    if z<beaker[2]+.015 and c[2]-radius<beaker[2]+.008 and c[2]+radius>cb.table_z and np.linalg.norm(c[:2]-beaker[:2])<.028+radius:
                        last_reason=(name,'beaker_sphere',str(model.m.geom(g).name));failed=True;break
                if failed:break
                p,R=model.pose(arm.palm);actual_mouth=p+R@local;d=R@np.array(scene['axis_local'])
                for center in actual_mouth-np.linspace(0,.15,12)[:,None]*d:
                    if center[2]-.0125<cb.table_z+.015:last_reason=(name,'tube_table');failed=True;break
                    if center[2]-.0125<beaker[2]+.005 and center[2]+.0125>cb.table_z and np.linalg.norm(center[:2]-beaker[:2])<.0375:
                        last_reason=(name,'tube_beaker');failed=True;break
                if failed:break
                row=o.arm_cmd.copy();row[7:]=q;rows.append(row)
            if failed:break
            previous=opt.x.copy()
          if failed:break
          phases.append({'name':name,'rows':np.array(rows),'settle':settle,'mouth':np.array(points[-1][1]),'rotation':model.pose(arm.palm)[1].copy()})
        if failed:
            print('POUR_PLAN_REFUSED',H,final_angle,last_reason,flush=True);continue
        metadata={'height':H,'fixed_y':fixed_y,'free_roll':free_roll,'final_z':final_z,'jet_profile':jet_profile,'jet_speed':jet_speed,'lip_radius':lip_radius,'first_angle':first_angle,'final_angle':final_angle,'offset':offset.tolist(),'min_robot_z':minimum,'max_pose_residual':maximum_error,'phase_steps':[(p['name'],len(p['rows']),p['settle']) for p in phases]}
        print('POUR_PLAN',metadata,flush=True)
        return {'phases':phases,'metadata':metadata,'local_mouth':local,'beaker':beaker}
    print('NO_POUR_PLAN',last_reason,flush=True);return None
