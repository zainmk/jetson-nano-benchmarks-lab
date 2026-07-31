# parse_tegrastats.py  -> mean/peak power (W), peak RAM (MB), peak GPU %
# move/use on board to parse tegrastats logs
import re, sys, statistics as st
pw, ram, gpu = [], [], []
for line in open(sys.argv[1]):
    # Orin reports total input power as VDD_IN (mW). VERIFY this field in your log!
    m = re.search(r'VDD_IN (\d+)mW', line) or re.search(r'VDD_IN (\d+)/\d+', line)
    if m: pw.append(int(m.group(1))/1000.0)          # mW -> W
    r = re.search(r'RAM (\d+)/(\d+)MB', line)
    if r: ram.append(int(r.group(1)))
    g = re.search(r'GR3D_FREQ (\d+)%', line)
    if g: gpu.append(int(g.group(1)))
print(f"mean_power_W={st.mean(pw):.2f} peak_power_W={max(pw):.2f} "
      f"peak_ram_MB={max(ram)} peak_gpu_pct={max(gpu) if gpu else 'NA'}")