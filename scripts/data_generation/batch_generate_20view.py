import os, sys, glob, json, time, argparse
import multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from data_generation.Vascular_PreTraining_Data import process_single_mesh

def worker(item):
    stl_path, save_root, ds_name = item
    try:
        ok, msg = process_single_mesh(
            stl_path=stl_path,
            save_root=save_root,
            dataset_name=ds_name,
            n_random_walks=20,
            base_walks=20,
            perturb_sigma=0.05,
            geometry_backend="auto",
            collision_backend="fcpw_ray",
            save_dtype="float32",
            force=True
        )
        return ok, msg, stl_path
    except Exception as e:
        return False, str(e), stl_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl_root", type=str, default="./HemoData/Vascular_STL")
    parser.add_argument("--save_root", type=str, default="./HemoData/Vascular_PreTrain")
    parser.add_argument("--datasets", nargs="+", default=["4TCTA_AAA", "aneumo", "IntrA", "AneuRisk", "CMHA", "VMR"])
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_per_dataset", type=int, default=0)
    args = parser.parse_args()

    print(f"Starting 20-view generation on datasets: {args.datasets}")
    print(f"Source STL root: {args.stl_root}")
    print(f"Destination root: {args.save_root}")
    print(f"Parallel workers: {args.num_workers}")

    tasks = []
    for ds in args.datasets:
        ds_stl_dir = os.path.join(args.stl_root, ds)
        if not os.path.exists(ds_stl_dir):
            print(f"Warning: {ds_stl_dir} does not exist, skipping.")
            continue
        stl_files = sorted(glob.glob(os.path.join(ds_stl_dir, "*.stl")))
        if args.max_per_dataset > 0:
            stl_files = stl_files[:args.max_per_dataset]
        print(f"Dataset {ds}: found {len(stl_files)} STL files to process.")
        for stl in stl_files:
            tasks.append((stl, args.save_root, ds))

    print(f"Total meshes to process: {len(tasks)}")
    t0 = time.time()

    success_cnt = 0
    fail_cnt = 0
    with mp.Pool(args.num_workers) as pool:
        for ok, msg, stl_path in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="Generating 20-view data"):
            if ok:
                success_cnt += 1
            else:
                fail_cnt += 1
                print(f"Failed {stl_path}: {msg}")

    total_time = time.time() - t0
    print(f"\nCompleted 20-view generation in {total_time:.2f}s!")
    print(f"Success: {success_cnt}, Failed: {fail_cnt}")

if __name__ == "__main__":
    main()
