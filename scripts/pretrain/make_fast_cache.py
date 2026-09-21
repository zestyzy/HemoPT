import os, json, time, sys

root = sys.argv[1] if len(sys.argv) > 1 else 'HemoData/Vascular_PreTrain'
out_file = sys.argv[2] if len(sys.argv) > 2 else 'data_provider/samples_index_fast.json'

print(f"Scanning subdirectories under {root}...")
subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
print("Found subdirectories:", subdirs)

discovered = []
for sd in subdirs:
    sd_path = os.path.join(root, sd)
    samples = sorted(os.listdir(sd_path))
    valid_in_sub = 0
    for s in samples:
        s_dir = os.path.join(sd_path, s)
        if not os.path.isdir(s_dir):
            continue
        if os.path.exists(os.path.join(s_dir, 'x.npy')):
            discovered.append(os.path.abspath(s_dir))
            valid_in_sub += 1
    print(f"  {sd}: {valid_in_sub} valid samples")

print(f"Total valid samples: {len(discovered)}")
with open(out_file, 'w') as f:
    json.dump(discovered, f)
print(f"Saved cache to {out_file}")
