import os,json,time,hashlib,base64,ctypes,shutil,psutil
from pathlib import Path
ROOT=Path("E:/gpu_training_scratchpad")
X=ROOT/"analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1"
OUT=X/"r8_race500k_completion_actual04_v1"
EXT=X/"r8_race500k_ext_04"
BODY=X/"r8_race500k_04"
DIRECT=X/"r8_race500k_direct_numeric_actual04_v1.json"
SEED_B64="eyJhY3R1YWxfcHVibGljX2VudHJ5X2luaXRpYWxfRlVMTF9UT09MIjp7ImNodW5rX2lkIjoiNTBkMDEwIiwid2FsbF90aW1lX3NlY29uZHMiOjEwLjAwMzQzMzUsInNlc3Npb25faWQiOjIzNDExLCJvcmlnaW5hbF90b2tlbl9jb3VudCI6MCwib3V0cHV0IjoiIn0sImFjdHVhbF9wdWJsaWNfZW50cnlfaW50ZXJtZWRpYXRlX0ZVTExfVE9PTCI6W3siY2h1bmtfaWQiOiJkNWU5N2QiLCJ3YWxsX3RpbWVfc2Vjb25kcyI6MTAuMDA0Njc5NSwic2Vzc2lvbl9pZCI6MjM0MTEsIm9yaWdpbmFsX3Rva2VuX2NvdW50IjowLCJvdXRwdXQiOiIifSx7ImNodW5rX2lkIjoiYzY3NGJjIiwid2FsbF90aW1lX3NlY29uZHMiOjMwLjAxNzkxMTksInNlc3Npb25faWQiOjIzNDExLCJvcmlnaW5hbF90b2tlbl9jb3VudCI6MCwib3V0cHV0IjoiIn1dLCJhY3R1YWxfcHVibGljX2VudHJ5X3Rlcm1pbmFsX0ZVTExfVE9PTCI6eyJjaHVua19pZCI6IjU2M2M5NCIsIndhbGxfdGltZV9zZWNvbmRzIjowLjAwMDAwNywiZXhpdF9jb2RlIjoxLCJvcmlnaW5hbF90b2tlbl9jb3VudCI6NzU1LCJvdXRwdXQiOiJUcmFjZWJhY2sgKG1vc3QgcmVjZW50IGNhbGwgbGFzdCk6XHJcbiAgRmlsZSBcIkU6XFxncHVfdHJhaW5pbmdfc2NyYXRjaHBhZFxcYW5hbHlzaXNcXEJQRTkyMjNfdHJhaW5pbmdfbGF1bmNoX3YxXFxyZXNpZGVudF9SQU1fcmVjdXJyZW50X21ldGFkYXRhX3Nsb3dkb3duX2ZpeF92MVxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcbmV4dDIwMDAwMF90aHJvdWdocHV0X3JlcGFpcl92MVxcbmV4dDIxMjAwMF9mYWlsdXJlX3JlcGFpcl92MVxccjhfcHJvb2ZfcmFjZV92NS5weVwiLCBsaW5lIDEyMDcsIGluIDxtb2R1bGU+XHJcbiAgICBjb250cm9sbGVyKGFyZ3MuUk9PVF9yZXZpZXcpXHJcbiAgRmlsZSBcIkU6XFxncHVfdHJhaW5pbmdfc2NyYXRjaHBhZFxcYW5hbHlzaXNcXEJQRTkyMjNfdHJhaW5pbmdfbGF1bmNoX3YxXFxyZXNpZGVudF9SQU1fcmVjdXJyZW50X21ldGFkYXRhX3Nsb3dkb3duX2ZpeF92MVxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcbmV4dDIwMDAwMF90aHJvdWdocHV0X3JlcGFpcl92MVxcbmV4dDIxMjAwMF9mYWlsdXJlX3JlcGFpcl92MVxccjhfcHJvb2ZfcmFjZV92NS5weVwiLCBsaW5lIDExOTQsIGluIGNvbnRyb2xsZXJcclxuICAgIHJldHVybiBzdXBlcnZpc29yLmV4ZWN1dGUoc2NvcGVwYXRoLFBhdGgoX19maWxlX18pLnJlc29sdmUoKSxcInJlYWRlclwiKVxyXG4gICAgICAgICAgIF5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXlxyXG4gIEZpbGUgXCJFOlxcZ3B1X3RyYWluaW5nX3NjcmF0Y2hwYWRcXGFuYWx5c2lzXFxCUEU5MjIzX3RyYWluaW5nX2xhdW5jaF92MVxccmVzaWRlbnRfUkFNX3JlY3VycmVudF9tZXRhZGF0YV9zbG93ZG93bl9maXhfdjFcXG9uZV9wb3N0REVWX2djX3JlZnJlc2hfdjFcXG5leHQyMDAwMDBfdGhyb3VnaHB1dF9yZXBhaXJfdjFcXG5leHQyMTIwMDBfZmFpbHVyZV9yZXBhaXJfdjFcXG93bmVkX3NvdXJjZV9DUFVfcHJvb2Zfd2F0Y2hkb2dfdjEucHlcIiwgbGluZSAxMTEsIGluIGV4ZWN1dGVcclxuICAgIGNhcHR1cmUoKTt0aW1lLnNsZWVwKC4xKVxyXG4gICAgXl5eXl5eXl5eXHJcbiAgRmlsZSBcIkU6XFxncHVfdHJhaW5pbmdfc2NyYXRjaHBhZFxcYW5hbHlzaXNcXEJQRTkyMjNfdHJhaW5pbmdfbGF1bmNoX3YxXFxyZXNpZGVudF9SQU1fcmVjdXJyZW50X21ldGFkYXRhX3Nsb3dkb3duX2ZpeF92MVxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcbmV4dDIwMDAwMF90aHJvdWdocHV0X3JlcGFpcl92MVxcbmV4dDIxMjAwMF9mYWlsdXJlX3JlcGFpcl92MVxcb3duZWRfc291cmNlX0NQVV9wcm9vZl93YXRjaGRvZ192MS5weVwiLCBsaW5lIDczLCBpbiBjYXB0dXJlXHJcbiAgICBpZiBzYW1lKGlkZW50aXR5KTpzZWVkcy5hcHBlbmQocHN1dGlsLlByb2Nlc3MoaWRlbnRpdHlbXCJwaWRcIl0pKVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIF5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5eXl5cclxuICBGaWxlIFwiQzpcXFVzZXJzXFxhcmRhb1xcRG9jdW1lbnRzXFxDb2RleFxcMjAyNi0xMC0xMFxcaS14MjBcXHdvcmtcXG1vbml0b3ItZW52XFxMaWJcXHNpdGUtcGFja2FnZXNcXHBzdXRpbFxcX19pbml0X18ucHlcIiwgbGluZSAzMTQsIGluIF9faW5pdF9fXHJcbiAgICBzZWxmLl9pbml0KHBpZClcclxuICBGaWxlIFwiQzpcXFVzZXJzXFxhcmRhb1xcRG9jdW1lbnRzXFxDb2RleFxcMjAyNi0xMC0xMFxcaS14MjBcXHdvcmtcXG1vbml0b3ItZW52XFxMaWJcXHNpdGUtcGFja2FnZXNcXHBzdXRpbFxcX19pbml0X18ucHlcIiwgbGluZSAzNjAsIGluIF9pbml0XHJcbiAgICByYWlzZSBOb1N1Y2hQcm9jZXNzKHBpZCwgbXNnPW1zZykgZnJvbSBOb25lXHJcbnBzdXRpbC5Ob1N1Y2hQcm9jZXNzOiBwcm9jZXNzIFBJRCBub3QgZm91bmQgKHBpZD0yNDk2NClcclxue1wicGluXCI6e1wicGF0aFwiOlwiRTpcXFxcZ3B1X3RyYWluaW5nX3NjcmF0Y2hwYWRcXFxcYW5hbHlzaXNcXFxcQlBFOTIyM190cmFpbmluZ19sYXVuY2hfdjFcXFxccmVzaWRlbnRfUkFNX3JlY3VycmVudF9tZXRhZGF0YV9zbG93ZG93bl9maXhfdjFcXFxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcXFxuZXh0MjAwMDAwX3Rocm91Z2hwdXRfcmVwYWlyX3YxXFxcXG5leHQyMTIwMDBfZmFpbHVyZV9yZXBhaXJfdjFcXFxccjhfcmFjZTUwMGtfZGlyZWN0X251bWVyaWNfYWN0dWFsMDRfdjEuanNvblwiLFwiYnl0ZXNcIjo3NzAsXCJzaGEyNTZcIjpcIjAwZjM5NTY2NzMxMzE2NTZiZGI2YzE0MDI5Mzc2NjI3OWMxNzQ1NGFmYWI4N2U0Y2I5NjgzYzllMjRiNWNjZWNcIn0sXCJ2YWx1ZVwiOntcImltbWVkaWF0ZV9tb25pdG9yX1B5dGhvbl9MQVNURVhJVENPREVcIjoxLFwiYWN0dWFsX3B1YmxpY19hcmd2XCI6W1wiQzpcXFxcVXNlcnNcXFxcYXJkYW9cXFxcRG9jdW1lbnRzXFxcXENvZGV4XFxcXDIwMjYtMTAtMTBcXFxcaS14MjBcXFxcd29ya1xcXFxtb25pdG9yLWVudlxcXFxTY3JpcHRzXFxcXHB5dGhvbi5leGVcIixcIi1YXCIsXCJ1dGY4XCIsXCItQlwiLFwiRTpcXFxcZ3B1X3RyYWluaW5nX3NjcmF0Y2hwYWRcXFxcYW5hbHlzaXNcXFxcQlBFOTIyM190cmFpbmluZ19sYXVuY2hfdjFcXFxccmVzaWRlbnRfUkFNX3JlY3VycmVudF9tZXRhZGF0YV9zbG93ZG93bl9maXhfdjFcXFxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcXFxuZXh0MjAwMDAwX3Rocm91Z2hwdXRfcmVwYWlyX3YxXFxcXG5leHQyMTIwMDBfZmFpbHVyZV9yZXBhaXJfdjFcXFxccjhfcHJvb2ZfcmFjZV92NS5weVwiLFwiLS1ST09ULXJldmlld1wiLFwiRTpcXFxcZ3B1X3RyYWluaW5nX3NjcmF0Y2hwYWRcXFxcYW5hbHlzaXNcXFxcQlBFOTIyM190cmFpbmluZ19sYXVuY2hfdjFcXFxccmVzaWRlbnRfUkFNX3JlY3VycmVudF9tZXRhZGF0YV9zbG93ZG93bl9maXhfdjFcXFxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcXFxuZXh0MjAwMDAwX3Rocm91Z2hwdXRfcmVwYWlyX3YxXFxcXG5leHQyMTIwMDBfZmFpbHVyZV9yZXBhaXJfdjFcXFxcUk9PVF9yOF9yYWNlNTAwa18wMDE1X3YxLmpzb25cIl0sXCJ0aW1lXCI6MTc5Mjk3NDI0Ny40MjcsXCJraW5kXCI6XCJvd25lcl9ORVdfcjhfcmFjZTUwMGtfcHVibGljX2VudHJ5X2ltbWVkaWF0ZV9QeXRob25fZXhpdF92MVwiLFwic2hlbGxfcGlkXCI6MjE4NjR9LFwicmVjb3JkXCI6XCJvd25lcl9pbW1lZGlhdGVfbW9uaXRvcl9udW1lcmljXCJ9XHJcbiJ9LCJhY3R1YWxfZGlyZWN0X251bWVyaWNfcmV0YWluZWQiOnsicGluIjp7InBhdGgiOiJFOlxcZ3B1X3RyYWluaW5nX3NjcmF0Y2hwYWRcXGFuYWx5c2lzXFxCUEU5MjIzX3RyYWluaW5nX2xhdW5jaF92MVxccmVzaWRlbnRfUkFNX3JlY3VycmVudF9tZXRhZGF0YV9zbG93ZG93bl9maXhfdjFcXG9uZV9wb3N0REVWX2djX3JlZnJlc2hfdjFcXG5leHQyMDAwMDBfdGhyb3VnaHB1dF9yZXBhaXJfdjFcXG5leHQyMTIwMDBfZmFpbHVyZV9yZXBhaXJfdjFcXHI4X3JhY2U1MDBrX2RpcmVjdF9udW1lcmljX2FjdHVhbDA0X3YxLmpzb24iLCJieXRlcyI6NzcwLCJzaGEyNTYiOiIwMGYzOTU2NjczMTMxNjU2YmRiNmMxNDAyOTM3NjYyNzljMTc0NTRhZmFiODdlNGNiOTY4M2M5ZTI0YjVjY2VjIn0sInZhbHVlIjp7ImltbWVkaWF0ZV9tb25pdG9yX1B5dGhvbl9MQVNURVhJVENPREUiOjEsImFjdHVhbF9wdWJsaWNfYXJndiI6WyJDOlxcVXNlcnNcXGFyZGFvXFxEb2N1bWVudHNcXENvZGV4XFwyMDI2LTEwLTEwXFxpLXgyMFxcd29ya1xcbW9uaXRvci1lbnZcXFNjcmlwdHNcXHB5dGhvbi5leGUiLCItWCIsInV0ZjgiLCItQiIsIkU6XFxncHVfdHJhaW5pbmdfc2NyYXRjaHBhZFxcYW5hbHlzaXNcXEJQRTkyMjNfdHJhaW5pbmdfbGF1bmNoX3YxXFxyZXNpZGVudF9SQU1fcmVjdXJyZW50X21ldGFkYXRhX3Nsb3dkb3duX2ZpeF92MVxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcbmV4dDIwMDAwMF90aHJvdWdocHV0X3JlcGFpcl92MVxcbmV4dDIxMjAwMF9mYWlsdXJlX3JlcGFpcl92MVxccjhfcHJvb2ZfcmFjZV92NS5weSIsIi0tUk9PVC1yZXZpZXciLCJFOlxcZ3B1X3RyYWluaW5nX3NjcmF0Y2hwYWRcXGFuYWx5c2lzXFxCUEU5MjIzX3RyYWluaW5nX2xhdW5jaF92MVxccmVzaWRlbnRfUkFNX3JlY3VycmVudF9tZXRhZGF0YV9zbG93ZG93bl9maXhfdjFcXG9uZV9wb3N0REVWX2djX3JlZnJlc2hfdjFcXG5leHQyMDAwMDBfdGhyb3VnaHB1dF9yZXBhaXJfdjFcXG5leHQyMTIwMDBfZmFpbHVyZV9yZXBhaXJfdjFcXFJPT1RfcjhfcmFjZTUwMGtfMDAxNV92MS5qc29uIl0sInRpbWUiOjE3OTI5NzQyNDcuNDI3LCJraW5kIjoib3duZXJfTkVXX3I4X3JhY2U1MDBrX3B1YmxpY19lbnRyeV9pbW1lZGlhdGVfUHl0aG9uX2V4aXRfdjEiLCJzaGVsbF9waWQiOjIxODY0fSwicmVjb3JkIjoib3duZXJfaW1tZWRpYXRlX21vbml0b3JfbnVtZXJpYyJ9LCJvcmRpbmFyeV9wcmVhZG1pc3Npb25fbnVtZXJpY19zdW1tYXJ5Ijp7IlRPT0xfY2h1bmsiOiIwZjJlM2QiLCJUT09MX2V4aXQiOjAsIm9yZGluYXJ5Ijp7InBoeXNpY2FsIjoxNjEwNDkxMDg0OCwiRV9mcmVlIjo2NDQ5NzU0Mjc1ODQsInRpbWUiOjE3OTI5NzQxNDQuMjI5LCJjb21taXQiOjEyNDcxMTczMTIwfSwiYWxsNzZfZXhhY3RfZGVhZCI6dHJ1ZSwiYm90aF9wYXRoc19hYnNlbnQiOnRydWUsInVuYm91bmRfcmFjZV9jdXJyZW50X1BJRCI6Mzc0MDQsInVuYm91bmRfcmFjZV9jdXJyZW50X1BJRF9hYnNlbnQiOnRydWUsInVuYm91bmRfc2Vjb25kX3JhY2VfY3VycmVudF9QSUQiOjIwMDUyLCJ1bmJvdW5kX3NlY29uZF9yYWNlX2N1cnJlbnRfUElEX2Fic2VudCI6dHJ1ZSwidW5ib3VuZF9yYWNlX2hpc3RvcmljYWxfaWRlbnRpdHlfZGVhdGhfcGFyZW50X29yX09TX05PVF9jbGFpbWVkIjp0cnVlLCJmdWxsX1RPT0xfcHJlc2VydmVkX2JlZm9yZV9wYXJzaW5nX2luX293bmVyX3JldGFpbmVkX21lbW9yeSI6dHJ1ZX0sInNjb3BlX0ZJUlNUX3Jlc3VsdF9waW4iOnsicGF0aCI6IkU6XFxncHVfdHJhaW5pbmdfc2NyYXRjaHBhZFxcYW5hbHlzaXNcXEJQRTkyMjNfdHJhaW5pbmdfbGF1bmNoX3YxXFxyZXNpZGVudF9SQU1fcmVjdXJyZW50X21ldGFkYXRhX3Nsb3dkb3duX2ZpeF92MVxcb25lX3Bvc3RERVZfZ2NfcmVmcmVzaF92MVxcbmV4dDIwMDAwMF90aHJvdWdocHV0X3JlcGFpcl92MVxcbmV4dDIxMjAwMF9mYWlsdXJlX3JlcGFpcl92MVxccjhfcmFjZTUwMGtfc2NvcGVfRklSU1RfMDAxNV92MVxcRklSU1Rfc2NvcGVfcmVhZF9yZXN1bHQuanNvbiIsImJ5dGVzIjoyODQxNSwic2hhMjU2IjoiZTFhMzk3ZjgwNTY2NGU3ZTY3MTQ4OTRmZWI0ZTAyMzJlMGNkYzEzZDM3NjBmOTA1MGM2NTliMTBiOWM2MDY2OSJ9LCJzY29wZV9GSVJTVF9SQVdfcGluIjp7InBhdGgiOiJFOlxcZ3B1X3RyYWluaW5nX3NjcmF0Y2hwYWRcXGFuYWx5c2lzXFxCUEU5MjIzX3RyYWluaW5nX2xhdW5jaF92MVxccmVzaWRlbnRfUkFNX3JlY3VycmVudF9tZXRhZGF0YV9zbG93ZG93bl9maXhfdjFcXG9uZV9wb3N0REVWX2djX3JlZnJlc2hfdjFcXG5leHQyMDAwMDBfdGhyb3VnaHB1dF9yZXBhaXJfdjFcXG5leHQyMTIwMDBfZmFpbHVyZV9yZXBhaXJfdjFcXHI4X3JhY2U1MDBrX3Njb3BlX0ZJUlNUXzAwMTVfdjFcXEZJUlNUX05FV19zY29wZV9SQVcuYmluIiwiYnl0ZXMiOjIzOTQwLCJzaGEyNTYiOiIyNmM1ZmMzZDZlYjU2YmZlN2FlZTEwOWY1MGQ3ZDNiOGE4ODMyZmViZGU4OTVkZTIwMDNjZjcxMTUzNDY4NWJmIn0sInNjb3BlX0ZJUlNUX21ldGFkYXRhX2lkZW50aXRpZXMiOlt7InBpZCI6MzMwODgsImNyZWF0ZV90aW1lIjoxNzkyOTc0MDYwLjAzMjQyNTIsInBwaWQiOjI4MjI0fSx7InBpZCI6MjgyMjQsImNyZWF0ZV90aW1lIjoxNzkyOTc0MDYwLjAyNTA2MiwicHBpZCI6MzgyOTZ9LHsicGlkIjozODI5NiwiY3JlYXRlX3RpbWUiOjE3OTI5NzQwNTkuNzU1NDk3MiwicHBpZCI6NjYyMH1dLCJzZXNzaW9uMjM0MTFfQ0xPU0VEX293bmVyX2NvbnN1bWVkX29uY2UiOnRydWUsIm5vX2FjdHVhbF9hY2NlcHRhbmNlX2NsYWltIjp0cnVlfQ=="
p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic()
class MEMORY(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in("totalphysical","physical","commitlimit","commit","totalvirtual","availablevirtual","extended")]
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def sample():
    m=MEMORY();m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(time=time.time(),pid=p.pid,ctime=p.create_time(),ppid=p.ppid(),commit=m.commit,physical=m.physical,private=p.memory_info().private,E_free=shutil.disk_usage(ROOT).free,elapsed=time.monotonic()-began)
def save(name,value):
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode("utf-8")
    path=OUT/name
    with path.open("xb")as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=sha(raw))
def limits(value,commit,physical):
    assert value["commit"]>=commit and value["physical"]>=physical and value["private"]<=536870912 and value["E_free"]>=100000000000 and value["elapsed"]<180,("Unchanged metadata resource predicate",value)
def guard():
    value=sample();limits(value,3509949440,3009949440);return value
OUT.mkdir(exist_ok=False)
first=sample()
initial=save("initial_BEFORE_predicate.json",dict(sample=first,limits=dict(prospective_commit=6028384896,prospective_physical=3278384896,runtime_commit=3509949440,runtime_physical=3009949440,self_private=536870912,wall=180,E_free=100000000000)))
fresh=sample();gate=save("fresh_numeric_BEFORE_predicate.json",dict(sample=fresh,initial=initial))
print(json.dumps(dict(record="numeric_BEFORE_predicate",first=first,fresh=fresh,initial=initial,gate=gate),separators=(",",":")),flush=True)
limits(first,6028384896,3278384896);limits(fresh,6028384896,3278384896)
for name in("TEMP","TMP","TORCH_EXTENSIONS_DIR"):
    assert os.environ.get(name)and Path(os.environ[name]).resolve().is_relative_to(ROOT.resolve())
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
assert all(os.environ.get(name)=="1"for name in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"))
authority=save("seal_scope_BEFORE_NEW_tree_or_artifact_stats.json",dict(kind="owner_NEW_metadata_live_order_500k_completion_scalar_FIRST_metadata_scope_v1",accepted=True,reviewer="/root",owner="/root/memory_diagnosis",proof_scope_SHA256="26c5fc3d6eb56bfe7aee109f50d7d3b8a8832febde895de2003cf711534685bf",actual_one_attempt_terminal=True,metadata_only=True,maximum_tree_files=256,maximum_depth=6,maximum_selected_files=32,maximum_each_JSON_bytes=1048576,maximum_each_log_bytes=16777216,maximum_selected_total_bytes=4194304,identity_union_maximum=32,only="NEW owned scalar witnesses/results/initials/Popen/guard/numeric/method/STATEfailure-replay/childentry/logs; ALL RAW before ANY decode; no fixture DB/history/journalpayload/liveReader7/Source/oldscope reads; no control",initial=initial,gate=gate))
seedraw=SEED_B64.encode("ascii")
seedrawpath=OUT/"retained_SEED_B64_RAW.bin"
with seedrawpath.open("xb")as stream:
    stream.write(seedraw);stream.flush();os.fsync(stream.fileno())
seedrawpin=dict(path=str(seedrawpath.resolve()),bytes=len(seedraw),sha256=sha(seedraw))
seedrawindex=save("retained_SEED_B64_RAW_index_BEFORE_base64_JSON_decode.json",dict(RAW=seedrawpin,RAW_fsynced_before_base64_or_JSON_decode=True))
seed=json.loads(base64.b64decode(seedraw))
seedpin=save("retained_NEW_public_entry_FULL_TOOL_and_preadmission.json",seed)
mandatory={DIRECT,EXT/"initial_BEFORE_full_predicate.json",EXT/"actual_Popen_identity.json",EXT/"external_result.json",EXT/"fixture_stdout.log",EXT/"fixture_stderr.log",BODY/"initial_BEFORE_predicate.json",BODY/"result.json"}
paths=set(mandatory);inventory=[];forced=[];files_seen=0
explicit_names={"concurrent_stage_ownership_witness.json","focused_STATE_before_assert.json","focused_STATE_replay_before_close.json","cold_burst_requests.json","response_loss_fence_assertion.json","response_loss_fence_release.json","actual_fixture_DB_path_binding.json","Windows_STATE_before_assert.json","status_polling_open_witness.json","mutable_JSON_observer_witness.json","STATE_failure_fence_assertion.json","response_loss_observed.json","child_entry.json","child_method_closure.json"}
def selected_scalar(name):
    lower=name.lower()
    if name in explicit_names:return True
    if lower.startswith(("request_lifecycle_","headroom_first_guard_"))and lower.endswith(".json"):return True
    if lower.startswith("forced_cleanup_")and lower.endswith(".json"):return True
    if lower.endswith(".json"):
        return any(word in lower for word in("receipt","witness","closure","result","failure","negative","observed","identity","child_entry","childentry","popen","initial"))
    return lower.endswith(".log")and any(word in lower for word in("stdout","stderr"))
def scalar_role(path):
    name=path.name;lower=name.lower()
    if path in mandatory:return(0,"mandatory_core")
    if lower.startswith("forced_cleanup_")and lower.endswith(".json"):return(1,"forced_cleanup")
    if name=="mutable_JSON_observer_witness.json":return(1,"mutable_JSON_observer_witness_mandatory_if_present")
    if name=="child_entry.json":return(2,"child_entry")
    if name=="child_method_closure.json":return(2,"child_method_closure")
    if name=="status_polling_open_witness.json":return(3,"status_polling_open_witness")
    if name in("Windows_STATE_before_assert.json","focused_STATE_before_assert.json"):return(4,"STATE_before_assert")
    if name=="focused_STATE_replay_before_close.json":return(4,"STATE_replay_before_close")
    if name=="STATE_failure_fence_assertion.json":return(4,"STATE_failure_fence_assertion")
    if name=="response_loss_observed.json":return(4,"response_loss_observed")
    if name=="response_loss_fence_assertion.json":return(4,"response_loss_fence_assertion")
    if name=="response_loss_fence_release.json":return(4,"response_loss_fence_release")
    if name=="cold_burst_requests.json":return(5,"cold_burst_requests")
    if lower.startswith("request_lifecycle_"):return(6,"lifetime_"+name)
    if lower.startswith("headroom_first_guard_"):return(6,"headroom_"+name)
    if name=="actual_fixture_DB_path_binding.json":return(7,"fixture_DB_path_binding")
    if "state"in lower:return(8,"STATE_auxiliary_scalar")
    if "lifetime"in lower:return(8,"lifetime_auxiliary_scalar")
    return(9,"other_scalar_or_log")
for root in(EXT,BODY):
    guard()
    if not root.exists():
        inventory.append(dict(root=str(root),exists=False));continue
    for parent,dirs,names in os.walk(root,followlinks=False):
        guard();folder=Path(parent)
        assert len(folder.relative_to(root).parts)<=6 and len(dirs)<=32
        for name in dirs:
            child=folder/name
            assert not child.is_symlink()and child.resolve().is_relative_to(root.resolve())
        for name in names:
            files_seen+=1;assert files_seen<=256
            path=folder/name
            assert not path.is_symlink()and path.resolve().is_relative_to(root.resolve())
            info=path.stat()
            is_forced=name.startswith("forced_cleanup_")and name.endswith(".json")
            if is_forced:forced.append(str(path.resolve()))
            candidate=path in mandatory or selected_scalar(name)
            if candidate:paths.add(path)
            inventory.append(dict(path=str(path.resolve()),bytes=info.st_size,mtime_ns=info.st_mtime_ns,candidate_scalar_or_log=candidate,selected_scalar_or_log=False,unselected_content_unread=True))
candidate_stats=[]
for path in sorted(paths,key=str):
    guard();rank,role=scalar_role(path)
    if not path.exists():
        candidate_stats.append(dict(path=str(path.resolve()),exists=False,rank=rank,role=role));continue
    info=path.stat()
    candidate_stats.append(dict(path=str(path.resolve()),exists=True,bytes=info.st_size,mtime_ns=info.st_mtime_ns,maximum=16777216 if path.suffix==".log"else 1048576,rank=rank,role=role))
allstatpin=save("ALL_NEW_candidate_scalar_PRESTAT_BEFORE_SELECTION_OR_CONTENT.json",dict(inputs=candidate_stats,maximum_selected_files=32,maximum_selected_total_bytes=4194304,all_candidate_contents_unread=True))
chosen=[];chosen_paths=set();total=0
def rejection(value):
    if value["exists"]and value["bytes"]>value["maximum"]:return"per_file_metadata_bound"
    if len(chosen)>=32:return"selected_file_count32"
    if value["exists"]and total+value["bytes"]>4194304:return"selected_total4MiB"
    return None
def choose(value):
    global total
    reason=rejection(value)
    if reason is not None:return reason
    chosen.append(value);chosen_paths.add(value["path"])
    if value["exists"]:total+=value["bytes"]
    return None
critical=sorted((value for value in candidate_stats if value["rank"]<=1),key=lambda value:(value["rank"],value["path"]))
for value in critical:
    reason=choose(value)
    if reason is not None:
        save("MANDATORY_metadata_selection_HOLD_BEFORE_ANY_content.json",dict(reason=reason,blocked=value,all_candidate_PRESTAT=allstatpin,already_selected=chosen,selected_bytes=total,all_original_contents_unread=True,automatic_retry_authorized=False))
        raise RuntimeError("Mandatory completion scalar metadata cannot fit unchanged32/4MiB selection: "+reason)
roles=sorted({(value["rank"],value["role"])for value in candidate_stats if value["rank"]>1 and value["exists"]})
for rank,role in roles:
    members=sorted((value for value in candidate_stats if value["rank"]==rank and value["role"]==role and value["exists"]),key=lambda value:value["path"])
    for value in members:
        if choose(value)is None:break
for value in sorted(candidate_stats,key=lambda value:(value["rank"],value["role"],value["path"])):
    if value["path"]not in chosen_paths:choose(value)
omitted=[]
for value in candidate_stats:
    if value["path"]not in chosen_paths:
        omitted.append(dict(STAT_only=value,reason=rejection(value)or"priority_selection",content_and_identity_facts="UNKNOWN_UNREAD"))
coverage=[]
for rank,role in sorted({(value["rank"],value["role"])for value in candidate_stats}):
    members=[value for value in candidate_stats if value["rank"]==rank and value["role"]==role and value["exists"]]
    selected=[value["path"]for value in members if value["path"]in chosen_paths]
    coverage.append(dict(rank=rank,role=role,present_count=len(members),selected_count=len(selected),selected_paths=selected,unselected_contents_and_identity_facts_unknown=len(selected)<len(members)))
for value in inventory:
    if "path"in value:
        value["selected_scalar_or_log"]=value["path"]in chosen_paths
        value["unselected_content_unread"]=not value["selected_scalar_or_log"]
assert len(chosen)<=32 and total<=4194304
inventorypin=save("NEW_owned_tree_STAT_selection_and_forced_cleanup_filename_observation.json",dict(inventory=inventory,all_candidate_PRESTAT=allstatpin,role_coverage=coverage,STAT_only_unselected_candidates=omitted,forced_cleanup_paths_observed=forced,forced_cleanup_artifacts_mandatory=True,required_core_scalar_and_full_fixture_logs_mandatory=True,status_polling_and_child_identity_method_scalar_roles_explicit=True,fixture_DB_history_journalpayload_and_Source_contents_unread=True,absence_is_current_complete_bounded_filename_observation_only=True))
prestats=sorted(chosen,key=lambda value:value["path"])
prestatpin=save("ALL_selected_NEW_scalar_log_PRESTAT_BEFORE_ANY_content.json",dict(inputs=prestats,total_bytes=total,maximum_total=4194304,maximum_files=32,STAT_only_unselected_manifest=inventorypin))
captures=[];raw_memory={}
for value in prestats:
    guard()
    if not value["exists"]:
        captures.append(value);continue
    path=Path(value["path"]);raw=path.read_bytes()
    rawpath=OUT/("FIRST_NEW_scalar_log_%03d_RAW.bin"%len(captures))
    with rawpath.open("xb")as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pin=dict(path=str(path.resolve()),bytes=len(raw),sha256=sha(raw))
    binary=dict(path=str(rawpath.resolve()),bytes=len(raw),sha256=sha(raw))
    after=path.stat()
    assert len(raw)==value["bytes"]==after.st_size and after.st_mtime_ns==value["mtime_ns"]
    captures.append(dict(pin=pin,RAW_binary=binary,RAW_fsynced_before_decode_or_expectedpin=True))
    raw_memory[str(path.resolve())]=raw
capturepin=save("ALL_NEW_scalar_log_RAW_capture_index_BEFORE_any_decode_JSON_expectedpin.json",dict(inputs=captures,prestat=prestatpin,inventory=inventorypin,all_selected_original_bytes_RAW_fsynced_before_any_decode_JSON_expectedpin=True))
direct_key=str(DIRECT.resolve());expected_direct=seed["actual_direct_numeric_retained"]["pin"]
if direct_key in raw_memory:
    assert dict(path=direct_key,bytes=len(raw_memory[direct_key]),sha256=sha(raw_memory[direct_key]))==expected_direct
texts={};objects={};decoded=[]
for path,raw in raw_memory.items():
    guard();text=raw.decode("utf-8-sig");texts[path]=text
    if path.endswith(".json"):objects[path]=json.loads(text)
    decoded.append(save("FIRST_NEW_scalar_log_%03d_UTF8_copy.json"%len(decoded),dict(pin=dict(path=path,bytes=len(raw),sha256=sha(raw)),UTF8_text=text)))
decodedpin=save("ALL_NEW_decoded_scalar_log_copies_AFTER_ALL_RAW.json",dict(copies=decoded,RAW_index=capturepin))
external=objects.get(str((EXT/"external_result.json").resolve()))
body=objects.get(str((BODY/"result.json").resolve()))
identities={};missing_time=[];nodes=0
def visit(value,origin):
    global nodes
    nodes+=1;assert nodes<=200000
    if isinstance(value,dict):
        if type(value.get("pid"))is int:
            created=value.get("ctime",value.get("create_time"))
            if type(created)in(int,float):
                key=(value["pid"],created)
                recorded_ppid=value.get("ppid")
                item=identities.get(key)
                if item is None:
                    item=dict(pid=value["pid"],ctime=created,ppid=recorded_ppid,origins=[])
                    identities[key]=item
                else:
                    if item["ppid"]is not None and recorded_ppid is not None:
                        assert item["ppid"]==recorded_ppid,("Conflicting explicit recorded parents",item,value,origin)
                    elif item["ppid"]is None and recorded_ppid is not None:
                        item["ppid"]=recorded_ppid
                if origin not in item["origins"]:item["origins"].append(origin)
            elif len(missing_time)<256:
                missing_time.append(dict(pid=value["pid"],origin=origin,ctime="UNKNOWN"))
        for child in value.values():visit(child,origin)
    elif isinstance(value,list):
        for child in value:visit(child,origin)
for path,value in objects.items():
    if path!=direct_key:visit(value,path)
visit(seed["scope_FIRST_metadata_identities"],"retained_seed.scope_FIRST_metadata_identities_NONfixture_role")
assert len(identities)<=32
observations=[]
for value in identities.values():
    guard()
    try:
        process=psutil.Process(value["pid"])
        actual=dict(pid=process.pid,ctime=process.create_time(),ppid=process.ppid())
        same=actual["ctime"]==value["ctime"]
        observations.append(dict(recorded=value,actual=actual,exact_recorded_identity_dead=not same,same_recorded_creation_time_alive=same,OS_or_method_exit_NOT_inferred_from_death=True))
    except psutil.NoSuchProcess:
        observations.append(dict(recorded=value,current_PID_absent=True,exact_recorded_identity_dead=True,OS_or_method_exit_NOT_inferred_from_death=True))
closurepin=save("NEW_complete_recorded_fixture_identity_union_and_exact_observations.json",dict(bound32=True,identities=observations,recorded_PID_without_ctime=missing_time,all_recorded_exact_dead=all(item["exact_recorded_identity_dead"]for item in observations),extraction="Every nested JSON dictionary with integer pid plus numeric ctime/create_time in ALL selected scalar files, plus three retained scope-FIRST metadata identities without fixture-role claim; strict creation-time equality and consistent explicit recorded parents; no descendant-order role inference",unobserved_descendants_unknown=True,unselected_scalar_contents_and_identity_facts_unknown=True,unknown_I30_descendants_and_OS_exit_outside_cleanup=True))
own=[];process=p
for role in("completion_body","completion_parent1","completion_parent2"):
    if process is None:break
    try:
        own.append(dict(role=role,pid=process.pid,ctime=process.create_time(),ppid=process.ppid()))
        process=process.parent()
    except psutil.NoSuchProcess:
        own.append(dict(role=role,identity="UNAVAILABLE"));break
ownpin=save("completion_own_metadata_identity_chain.json",own)
stderr=texts.get(str((EXT/"fixture_stderr.log").resolve()),"")
completion=save("completion.json",dict(kind="owner_NEW_metadata_live_order_500k_exact_ONE_completion_scalar_seal_v1",independent_ROOT_acceptance_PENDING=True,scope=authority,initial=initial,gate=gate,retained_NEW_full_TOOL_and_preadmission=seedpin,retained_SEED_B64_RAW_before_decode=seedrawindex,actual_public_entry_TOOL_exit=seed["actual_public_entry_terminal_FULL_TOOL"]["exit_code"],actual_immediate_monitor_Python_LASTEXITCODE=seed["actual_direct_numeric_retained"]["value"]["immediate_monitor_Python_LASTEXITCODE"],capture_index=capturepin,decoded_copies=decodedpin,inventory=inventorypin,recorded_family_identity_union=closurepin,completion_own_metadata_identity_chain=ownpin,actual_external_result=external,actual_body_result=body,body_result_absent=body is None,actual_direct_numeric=objects.get(direct_key),forced_cleanup_artifacts_observed=forced,forced_cleanup_absence_is_current_filename_observation_only=True,empty_cleanup_array_alone_is_NOT_absence_evidence=True,fixture_stderr_tail=stderr[-12000:],no_normal_OS_or_method_closure_fact_inferred_from_TOOL_or_death=True,exact_observed_values_and_open_error_codes_remain_in_full_selected_witness_copies=True,old_scopes_and_actuals_stay_CLOSED_HOLD=True,no_further_proof_Source_control_or_production_authority=True,liveReader7_DB_history_journalpayload_Source_content_reads=0,control=0,final=guard()))
print(json.dumps(dict(record="NEW_metadata_live_order_500k_completion_metadata_seal",completion=completion,initial=initial,gate=gate,scope=authority,retained_TOOL=seedpin,retained_SEED_B64_RAW_before_decode=seedrawindex,capture_index=capturepin,decoded_copies=decodedpin,closure=closurepin,own=own,inputs=captures,forced_cleanup_paths_observed=forced,actual_public_entry_TOOL_exit=seed["actual_public_entry_terminal_FULL_TOOL"]["exit_code"],actual_immediate_monitor_Python_LASTEXITCODE=seed["actual_direct_numeric_retained"]["value"]["immediate_monitor_Python_LASTEXITCODE"],body_passed=body.get("passed")if isinstance(body,dict)else None,body_case_count=len(body.get("cases",[]))if isinstance(body,dict)else None,external_result=external,fixture_stderr_tail=stderr[-12000:],final=guard()),separators=(",",":")),flush=True)
