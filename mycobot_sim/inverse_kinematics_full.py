import mujoco, numpy as np, math, time, imageio, pathlib, textwrap
from typing import List

# ═════════════════════════════════════════════════════════════
# 1)  Quaternion & DH utils
# ═════════════════════════════════════════════════════════════
def dh_transform(a, alpha, d, theta):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0 ,     sa,     ca,    d],
                     [0 ,      0,      0,    1]], float)

def nquat(q): n=np.linalg.norm(q); return q/n if n>1e-10 else np.array([1,0,0,0])

def euler_to_quat(e):
    a,b,c=e*0.5; ca,cb,cg=np.cos([a,b,c]); sa,sb,sg=np.sin([a,b,c])
    return nquat(np.array([ca*cb*cg-sa*cb*sg,
                           ca*sb*cg+sa*sb*sg,
                           sa*sb*cg-ca*sb*sg,
                           ca*cb*sg+sa*cb*cg]))

def rot_to_quat(R):
    t=np.trace(R)
    if t>0:
        S=math.sqrt(t+1)*2; w=.25*S
        x=(R[2,1]-R[1,2])/S; y=(R[0,2]-R[2,0])/S; z=(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S=math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
        w=(R[2,1]-R[1,2])/S; x=.25*S; y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]:
        S=math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
        w=(R[0,2]-R[2,0])/S; x=(R[0,1]+R[1,0])/S; y=.25*S; z=(R[1,2]+R[2,1])/S
    else:
        S=math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
        w=(R[1,0]-R[0,1])/S; x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=.25*S
    return nquat(np.array([w,x,y,z]))

def qmul(q1,q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

def qerr(qc, qd):                 # 3-vec error
    qe=qmul(qd, np.array([qc[0],-qc[1],-qc[2],-qc[3]]))
    if qe[0]<0: qe=-qe
    return 2*qe[1:]

# ═════════════════════════════════════════════════════════════
# 2)  FK & Jacobian  (FK returns [x y z qw qx qy qz])
# ═════════════════════════════════════════════════════════════
def fk(q):
    t1,t2,t3,t4,t5,t6=q; t2-=np.pi/2; t4-=np.pi/2; t5+=np.pi/2
    dh=[(0,np.pi/2,0.15708,t1),(-0.1104,0,0,t2),(-0.096,0,0,t3),
        (0,np.pi/2,0.06639,t4),(0,-np.pi/2,0.07318,t5),(0,0,0.0456,t6)]
    T=np.eye(4)
    for a,al,d,th in dh: T=T@dh_transform(a,al,d,th)
    pos=T[:3,3]; quat=rot_to_quat(T[:3,:3])
    return np.concatenate([pos, quat])

def jacobian(q):
    ee=fk(q)[:3]
    t1,t2,t3,t4,t5,t6=q; t2-=np.pi/2; t4-=np.pi/2; t5+=np.pi/2
    dh=[(0,np.pi/2,0.15708,t1),(-0.1104,0,0,t2),(-0.096,0,0,t3),
        (0,np.pi/2,0.06639,t4),(0,-np.pi/2,0.07318,t5),(0,0,0.0456,t6)]
    T=np.eye(4); Ts=[T]
    for a,al,d,th in dh:
        T=T@dh_transform(a,al,d,th); Ts.append(T.copy())
    J=np.zeros((6,6))
    for i in range(6):
        z=Ts[i][:3,2]; p=Ts[i][:3,3]
        J[:3,i]=np.cross(z, ee-p); J[3:,i]=z
    return J

# ═════════════════════════════════════════════════════════════
# 3)  Differential IK  (pos+quat)
# ═════════════════════════════════════════════════════════════
def dls_pinv(J, lam=1e-2): return J.T@np.linalg.inv(J@J.T+(lam**2)*np.eye(6))

def ik_step(q, target, *, kg=0.3, lam=1e-2, vmax=1.0):
    pc,qc=fk(q)[:3], fk(q)[3:]
    pd,qd=target[:3], target[3:]
    err=np.hstack([pd-pc, kg*qerr(qc,qd)])
    dq=dls_pinv(jacobian(q),lam)@err
    m=np.abs(dq).max()
    if vmax>0 and m>vmax: dq*=vmax/m
    return dq

# ═════════════════════════════════════════════════════════════
# 4)  XML marker (位置+quat で設置)
# ═════════════════════════════════════════════════════════════
def marker_xml(pos, quat):
    x,y,z=pos; w,qx,qy,qz=quat
    return textwrap.dedent(f"""
      <body name="target_marker" pos="{x} {y} {z}" quat="{w} {qx} {qy} {qz}">
        <geom type="sphere" size="0.015" rgba="1 0 0 .7"/>
        <geom type="capsule" fromto="0 0 0 0.03 0 0" size="0.003" rgba="1 0 0 .9"/>
        <geom type="capsule" fromto="0 0 0 0 0.03 0" size="0.003" rgba="0 1 0 .9"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.003" rgba="0 0 1 .9"/>
      </body>
    """)

def patch_xml(src:str, pos, quat)->str:
    src_path=pathlib.Path(src)
    xml=src_path.read_text()
    idx=xml.rfind("</worldbody>")
    if idx==-1: raise RuntimeError("worldbody closing tag not found")
    out_text=xml[:idx]+marker_xml(pos,quat)+xml[idx:]
    out_path=src_path.parent/(src_path.stem+"_with_target.xml")
    out_path.write_text(out_text)
    return str(out_path)

# ═════════════════════════════════════════════════════════════
# 5)  MuJoCo helper
# ═════════════════════════════════════════════════════════════
def set_q(data, q): data.qpos[:6]=q

# ═════════════════════════════════════════════════════════════
# 6)  Main
# ═════════════════════════════════════════════════════════════
def main():
    ORG_XML="config/xml/mycobot_280jn_mujoco.xml"
    dt, steps, vmax = 0.02, 500, 1.0

    # ── initial & target (quat) ───────────────────────────────
    q0=np.random.uniform(-0.1,0.1,6)
    p_tgt=np.array([0.04926,-0.1852,0.0566])
    q_tgt=euler_to_quat(np.array([np.pi/2,0,0]))   # 例：Z 90°
    target=np.hstack([p_tgt, q_tgt])

    patched=patch_xml(ORG_XML, p_tgt, q_tgt)

    # ── load model ────────────────────────────────────────────
    model=mujoco.MjModel.from_xml_path(patched)
    model.opt.timestep=dt
    data=mujoco.MjData(model)

    W,H=1920,1088
    model.vis.global_.offwidth=W; model.vis.global_.offheight=H
    rend=mujoco.Renderer(model,H,W); rend.enable_shadows=True

    # ── run sim ───────────────────────────────────────────────
    set_q(data,q0); mujoco.mj_forward(model,data)
    frames:List[np.ndarray]=[]; qlog=[]
    q=q0.copy()
    for k in range(steps):
        t=time.time()
        q+=ik_step(q,target,kg=0.3,lam=1e-2,vmax=vmax)*dt
        set_q(data,q); mujoco.mj_forward(model,data)
        rend.update_scene(data)
        if k%2==0: frames.append(rend.render())
        qlog.append(q.copy())
        time.sleep(max(0,dt-(time.time()-t)))

    # ── save ──────────────────────────────────────────────────
    imageio.mimsave("robot_quat_ik.mp4", frames, fps=30,
                    quality=8, bitrate="24M")
    np.savez("robot_quat_ik_logs.npz", q=np.array(qlog), target=target)
    rend.close(); print("✓ robot_quat_ik.mp4 saved")

if __name__=="__main__": main()
