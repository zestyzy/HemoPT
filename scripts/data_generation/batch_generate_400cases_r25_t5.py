import os, sys, glob, json, time, random, argparse
import multiprocessing as mp
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from data_generation.Vascular_PreTraining_Data import process_single_mesh

def worker(item):
    stl_path, save_root, ds_name, qc_row = item
    try:
        ok, msg = process_single_mesh(
            stl_path=stl_path,
            save_root=save_root,
            dataset_name=ds_name,
            n_random_walks=25,
            base_walks=25,
            perturb_sigma=0.05,
            geometry_backend="auto",
            collision_backend="fcpw_ray",
            save_dtype="float32",
            force=False,
            qc_row=qc_row,
            walk_steps=5
        )
        return ok, msg, stl_path, ds_name
    except Exception as e:
        return False, str(e), stl_path, ds_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl_root", type=str, default="./HemoData/Vascular_STL")
    parser.add_argument("--save_root", type=str, default="./HemoData/Vascular_PreTrain_400_R25_T5")
    parser.add_argument("--qc_manifest", type=str, default="./HemoData/Vascular_STL_QC/qc_manifest.jsonl")
    parser.add_argument("--vmr_reserved", type=str, default="./HemoData/VMR_CFD_Splits/vmr_cfd_split.json")
    parser.add_argument("--aneumo_reserved", type=str, default="./HemoData/Aneumo_CFD_Splits/aneumo_cfd_split.json")
    parser.add_argument("--num_cases", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    print("====================================================================")
    print(f" Starting 400-Case Preprocessing (R=25 views, T=5 steps, float32)")
    print(f" Target directory: {args.save_root}")
    print(f" Workers: {args.num_workers} | Seed: {args.seed}")
    print("====================================================================")

    # Load QC map
    qc_map = {}
    if os.path.exists(args.qc_manifest):
        with open(args.qc_manifest) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    rel = r.get("relative_path")
                    if rel:
                        qc_map[rel] = r
                except Exception:
                    pass

    # Load reserved cases
    vmr_reserved = set()
    if os.path.exists(args.vmr_reserved):
        with open(args.vmr_reserved) as f:
            vmr_reserved = set(json.load(f).get("reserved_case_ids", []))

    aneumo_reserved = set()
    if os.path.exists(args.aneumo_reserved):
        with open(args.aneumo_reserved) as f:
            aneumo_reserved = set(json.load(f).get("reserved_case_ids", []))

    datasets = ["4TCTA_AAA", "aneumo", "AneuRisk", "Aneux", "CMHA", "IntrA", "Totalsegmentator", "VMR"]
    candidates = []

    for ds in datasets:
        ds_dir = os.path.join(args.stl_root, ds)
        if not os.path.isdir(ds_dir):
            continue
        stl_files = sorted(glob.glob(os.path.join(ds_dir, "*.stl")))
        for s in stl_files:
            base = os.path.basename(s)
            cid = os.path.splitext(base)[0]
            rel = f"{ds}/{base}"

            # Filter QC fail
            if rel in qc_map and qc_map[rel].get("status") == "fail":
                continue
            # Filter reserved
            if ds == "VMR" and any(r in cid for r in vmr_reserved):
                continue
            if ds == "aneumo" and cid in aneumo_reserved:
                continue

            candidates.append((s, ds, cid, qc_map.get(rel)))

    print(f"Found {len(candidates)} candidate meshes passing QC and reservation checks.")
    random.seed(args.seed)
    random.shuffle(candidates)
    selected = candidates[:args.num_cases]
    print(f"Selected {len(selected)} meshes.")
    ds_dist = Counter([x[1] for x in selected])
    print(f"Dataset distribution: {dict(ds_dist)}")

    # Save manifest
    manifest_path = os.path.join(args.save_root, "manifest_400cases_r25_t5.json")
    manifest_data = {
        "num_cases": len(selected),
        "seed": args.seed,
        "n_random_walks": 25,
        "base_walks": 25,
        "walk_steps": 5,
        "dtype": "float32",
        "dataset_distribution": dict(ds_dist),
        "cases": [
            {"dataset": ds, "case_id": cid, "stl_path": os.path.abspath(s)}
            for s, ds, cid, _ in selected
        ]
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    # Build tasks
    tasks = [(s, args.save_root, ds, qc_row) for s, ds, cid, qc_row in selected]

    t0 = time.time()
    success_cnt = 0
    fail_cnt = 0

    with mp.Pool(args.num_workers) as pool:
        for ok, msg, stl_path, ds in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="Processing meshes"):
            if ok:
                success_cnt += 1
            else:
                fail_cnt += 1
                print(f"[FAILED] {ds}/{os.path.basename(stl_path)}: {msg}")

    elapsed = time.time() - t0
    print("====================================================================")
    print(f" Preprocessing completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
    print(f" Success: {success_cnt}/{len(tasks)} | Failed: {fail_cnt}")
    print("====================================================================")

if __name__ == "__main__":
    main()
