from mcp_daw import song as S
U="uzu-drumkit/"; W="karoryfer.swirly-drums/Samples/"; M="karoryfer.meatbass/Samples/pizz/"
s=S.new_song(bpm=80,bars=32,key="C minor",swing=0.45)
s["sections"]=[{"name":n,"start_bar":a,"bars":b} for n,a,b in [("intro",0,4),("groove A",4,8),("groove B",12,8),("breakdown",20,4),("groove A2",24,8)]]
R=lambda a,b:list(range(a,b))
A=R(4,20)+R(24,32); ALL=R(0,32); B=R(12,20)+R(24,32)
def T(n,smp,steps,bars,g=0,pan=0,p=None):
    S.add_pattern_track(s,n,smp,steps,pitches=p,bars=bars,gain_db=g,pan=pan)
o=lambda *ix,v=1:[ (v if i in ix else 0) for i in range(16)]
T("kick",U+"bd/12_bd_mot4i.wav",[1,0,0,0,0,0,0,.5,0,0,.9,0,0,0,0,0],A,-1)
T("kick_soft",W+"cajon_kick/cajon_kick_vl7_rr1_front.wav",o(0,10,v=.6),R(20,24),-6)
T("snare",U+"sd/14_sd.wav",o(4,12),A,-3)
T("ghost",W+"snare_main/snare_hit_vl1_rr1_top.wav",o(7,15,v=.4),A,-10,.1)
T("rim",U+"rim/10_rim_switchangel.wav",o(4,12,v=.7),R(0,4),-8)
T("hats",W+"hat_tight/hh_tight_vl4_rr1.wav",[.8,.3,.6,.3]*4,ALL,-12,.3)
T("open_hat",U+"oh/10_oh_switchangel.wav",o(14,v=.5),B,-15,.35)
T("shaker",U+"sh/10_sh_switchangel.wav",[0,.5,0,.7]*4,R(12,24)+R(24,32),-16,-.35)
T("ride",W+"ride/ride_vl4_rr1.wav",o(0,8,v=.5),R(0,4),-18,-.2)
T("crash",W+"crash/crash_vl4_rr1.wav",o(0,v=.6),[4,12,24],-14,.2)
roots=[0,-4,3,-2]; tri=[[0,3,7],[-4,0,3],[3,7,10],[-2,2,5]]
for k,r in enumerate(roots):
    bars=[b for b in ALL if b%4==k]
    T(f"bass_{k}",M+"c2_vl4_rr1.wav",[1,0,0,0,0,0,.6,0,0,0,.8,0,0,.5,0,0],bars,-1,-.05,[r,0,0,0,0,0,r,0,0,0,r+12 if r<0 else r,0,0,r+7,0,0])
    for j,iv in enumerate(tri[k]):
        T(f"keys_{k}_{j}",M+"c3_vl1_rr1.wav",o(0,11,v=.45),[b for b in bars if b>=12],-13,[-.3,0,.3][j],[iv]*16)
mel=[(0,7),(3,10),(6,12),(10,10),(13,7)], [(2,3),(6,5),(10,7)], [(0,10),(4,7),(8,5),(12,3)], [(2,2),(6,0),(11,-2)]
for k,notes in enumerate(mel):
    st=[0]*16; pt=[0]*16
    for i,n in notes: st[i]=.7; pt[i]=n
    T(f"melody_{k}",M+"c3_vl4_rr1.wav",st,[b for b in R(12,20)+R(24,31) if b%4==k],-7,.15,pt)
S.add_loop_track(s,"break",U+"brk/10_break_amen_pprocessed.wav",1,R(24,32),gain_db=-16)
S.save(s,"songs/lofi_full.json")
