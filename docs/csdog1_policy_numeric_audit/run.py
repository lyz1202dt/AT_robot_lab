import numpy as np, onnxruntime as ort, yaml, json, subprocess, hashlib
from pathlib import Path
p=Path('/tmp/csdog1-numeric-audit'); deploy=Path('/home/lyz/桌面/cyberstartrainanddeploy/Cyberstar_rl_deploy'); cfg=deploy/'src/cyber1_edu_sdk/config/mnn_model_policy_exported3.yaml'
c=yaml.safe_load(cfg.read_text()); sar=yaml.safe_load(Path('/space2/Project/rl_sar/policy/csdog1/robot_lab/config.yaml').read_text())['csdog1/robot_lab']
rng=np.random.default_rng(20261019)
rows=[]; labels=[]
ref=np.array(sar['default_dof_pos'],np.float32)
for name,n,command in [('stand',60,[0,0,0]),('moving_random',500,None),('low_speed',90,[.1,0,0]),('moving',80,[.4,0,.1]),('release',65,[0,0,0]),('heading_hold',90,[.5,0,0]),('clip_stress',80,[.4,.2,.1])]:
 for j in range(n):
  q=rng.normal(size=4);q/=np.linalg.norm(q)
  cmd=np.array(command if command is not None else [rng.uniform(-1.5,1.5),rng.uniform(-1.5,1.5),rng.uniform(.1,1.5)])
  if name=='stand':q=np.array([1,0,0,0])
  gyro=rng.normal(0,.4,3);pos=ref+rng.normal(0,.15,12);vel=rng.normal(0,2,12)
  if name=='clip_stress':gyro*=10000;vel*=10000
  rows.append(np.r_[float(j==0 and name!='release'),cmd,q,gyro,pos,vel]);labels.append(name)
a=np.asarray(rows,np.float32);a.tofile(p/'states.bin')
subprocess.run([str(p/'audit'),str(cfg),str(p/'states.bin'),str(p/'results.bin')],check=True,stdout=(p/'mnn_runtime.log').open('w'))
r=np.fromfile(p/'results.bin',np.float32).reshape(-1,307);history=r[:,:282];raw=r[:,282:294];target=r[:,294:306];sar_phase=r[:,306]
sess=ort.InferenceSession('/home/lyz/桌面/cyberstartrainanddeploy/policy.onnx',providers=['CPUExecutionProvider']);iname=sess.get_inputs()[0].name
onnx=np.array([sess.run(None,{iname:x[None]})[0][0] for x in history])
error=np.abs(raw-onnx); clipped=np.clip(raw,-100,100);ideal=ref+clipped*.25
# rl_sar ComputeObservation algebra, same MNN phase/previous action to isolate
# command scaling, quaternion inverse rotation, term order and history rolling.
expected_hist=None;align=[];phase_err=[];target_err=[];prev=np.zeros(12,np.float32)
for i,row in enumerate(a):
 if row[0]:expected_hist=None;prev=np.zeros(12,np.float32)
 q=row[4:8];xyz=q[1:];v=np.array([0,0,-1],np.float32)
 gravity=v*(2*q[0]*q[0]-1)-2*q[0]*np.cross(xyz,v)+2*xyz*np.dot(xyz,v)
 frame=np.r_[history[i,-47:-45],row[8:11]*sar['ang_vel_scale'],gravity,row[1:4]*sar['commands_scale'],(row[11:23]-ref)*sar['dof_pos_scale'],row[23:35]*sar['dof_vel_scale'],prev].astype(np.float32)
 frame=np.clip(frame,-100,100)
 if expected_hist is None:expected_hist=np.tile(frame,(6,1))
 else:expected_hist=np.r_[expected_hist[1:],frame[None]]
 align.append(np.max(np.abs(expected_hist.flatten()-history[i])))
 phase_err.append(np.max(np.abs(history[i,-47:-45]-[np.sin(2*np.pi*sar_phase[i]),np.cos(2*np.pi*sar_phase[i])])))
 target_err.append(np.max(np.abs(target[i]-ideal[i])))
 prev=clipped[i]
align=np.array(align);phase_err=np.array(phase_err);target_err=np.array(target_err)
summary={
 'method':'Actual deployment MNNModel::Impl::compute CPU runtime (2 threads) versus ONNX Runtime CPU on identical observed 282-D histories; rl_sar observation algebra and actual StopPhaseClock; no robot connection.',
 'samples':len(a),'onnx_input':iname,'onnx_output':sess.get_outputs()[0].name,
 'hashes':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path('/home/lyz/桌面/cyberstartrainanddeploy/policy.onnx'),Path('/space2/Project/rl_sar/policy/csdog1/robot_lab/policy.onnx'),deploy/'src/cyber1_edu_sdk/policy/policy_exported3.mnn']},
 'conversion':{'all_finite':bool(np.isfinite(raw).all() and np.isfinite(onnx).all()),'max_abs_raw_action_error':float(error.max()),'mean_abs_raw_action_error':float(error.mean()),'p99_abs_raw_action_error':float(np.quantile(error,.99)),'max_abs_clipped_target_conversion_error_rad':float(np.max(abs(np.clip(raw,-100,100)-np.clip(onnx,-100,100)))*.25)},
 'segments':{},'config_checks':{'reference_equal':c['default_joint_pos']==sar['default_dof_pos'],'command_scale_equal':[c['obs_scales']['cmd_lin_vel']]*2+[c['obs_scales']['cmd_yaw']]==sar['commands_scale'],'kp_equal':c['kp']==sar['rl_kp'],'kd_equal':c['kd']==sar['rl_kd']}}
for name in dict.fromkeys(labels):
 mask=np.array(labels)==name
 summary['segments'][name]={'samples':int(mask.sum()),'max_raw_conversion_error':float(error[mask].max()),'max_history_error_same_phase_reference':float(align[mask].max()),'max_phase_feature_difference_vs_sar_clock':float(phase_err[mask].max()),'max_target_postprocessing_difference_rad':float(target_err[mask].max())}
# Relative conversion tolerance suitable for FP32 runtime differences; stress
# input cases are measured separately, including observation clipping.
summary['conversion']['allclose_atol_1e-4_rtol_1e-4']=bool(np.allclose(raw,onnx,atol=1e-4,rtol=1e-4))
(p/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
np.savez_compressed(p/'vectors.npz',states=a,labels=np.array(labels),mnn_history=history,mnn_action=raw,onnx_action=onnx,mnn_target=target,sar_phase=sar_phase)
