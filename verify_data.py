"""
Verify that all migrated data sources are loadable and internally consistent.
Run: python verify_data.py
"""
import sys
import os
import pickle
import numpy as np

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def ok(msg):
    print(f"  [OK] {msg}")

def warn(msg):
    print(f"  [WARN] {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    raw = os.path.join(base, "data", "raw")
    errors = []

    # ------------------------------------------------------------------ #
    # 1. Load pandas-dependent sources
    # ------------------------------------------------------------------ #
    try:
        import pandas as pd
    except ImportError:
        fail("pandas not installed -- run: pip install pandas openpyxl")
        sys.exit(1)

    section("1. GMWM structure Excel (subjBrainDataNarrGen_360plain.xlsx)")
    gmwm_path = os.path.join(base, "subjBrainDataNarrGen_360plain.xlsx")
    try:
        gmwm_df = pd.read_excel(gmwm_path, header=[0, 1, 2, 3, 4])
        n_subj, n_cols = gmwm_df.shape
        site_ids_gmwm = gmwm_df.iloc[:, 0].astype(str).tolist()
        ok(f"Loaded: {n_subj} subjects x {n_cols} columns")
        ok(f"Unique subjects: {len(set(site_ids_gmwm))}")
    except Exception as e:
        fail(f"Cannot load GMWM Excel: {e}")
        errors.append("gmwm_excel")

    section("2. Connectivity Excel (subjBrainDataConnGen_360plain.xlsx)")
    conn_xlsx_path = os.path.join(base, "subjBrainDataConnGen_360plain.xlsx")
    try:
        conn_raw = pd.read_excel(conn_xlsx_path, header=None, nrows=6)
        conn_ids_row = conn_raw.iloc[0, :].astype(str).tolist()
        conn_subject_ids = [x for x in conn_ids_row if x != "Site_ID" and x != "nan"]
        ok(f"Header rows loaded OK; found {len(conn_subject_ids)} subject columns")
    except Exception as e:
        fail(f"Cannot load connectivity Excel: {e}")
        errors.append("conn_excel")

    section("3. Demographics Excel (XXL_ACEcohort395.xlsx)")
    demo_path = os.path.join(base, "XXL_ACEcohort395.xlsx")
    try:
        demo_df = pd.read_excel(demo_path)
        ok(f"Loaded: {demo_df.shape[0]} rows x {demo_df.shape[1]} columns")
        ok(f"Columns: {list(demo_df.columns)}")
        cohort_counts = demo_df["Cohort"].value_counts(dropna=False)
        for k, v in cohort_counts.items():
            ok(f"  Cohort={k}: {v}")
        gender_counts = demo_df["Gender"].value_counts(dropna=False)
        for k, v in gender_counts.items():
            ok(f"  Gender={k}: {v}")
    except Exception as e:
        fail(f"Cannot load demographics Excel: {e}")
        errors.append("demo_excel")

    section("4. Legacy demographics CSV (data/raw/demographicsData.csv)")
    legacy_demo_path = os.path.join(raw, "demographicsData.csv")
    try:
        legacy_df = pd.read_csv(legacy_demo_path)
        ok(f"Loaded: {legacy_df.shape[0]} rows x {legacy_df.shape[1]} columns")
        ok(f"Columns: {list(legacy_df.columns)}")
        classify_counts = legacy_df["Classify"].value_counts().sort_index()
        for k, v in classify_counts.items():
            ok(f"  Classify={k}: {v} subjects")
    except Exception as e:
        fail(f"Cannot load legacy demographics: {e}")
        errors.append("legacy_demo")

    section("5. Legacy GMWM structure CSV (data/raw/subjBrainDataGMWMStructure.csv)")
    gmwm_csv_path = os.path.join(raw, "subjBrainDataGMWMStructure.csv")
    try:
        gmwm_csv = pd.read_csv(gmwm_csv_path)
        ok(f"Loaded: {gmwm_csv.shape[0]} rows x {gmwm_csv.shape[1]} columns")
    except Exception as e:
        fail(f"Cannot load GMWM structure CSV: {e}")
        errors.append("gmwm_csv")

    section("6. Legacy connectivity CSV (data/raw/subjBrainDataConnGen_360.csv)")
    conn_csv_path = os.path.join(raw, "subjBrainDataConnGen_360.csv")
    try:
        conn_csv = pd.read_csv(conn_csv_path, nrows=5)
        ok(f"Loadable; sample shape {conn_csv.shape}")
    except Exception as e:
        fail(f"Cannot load connectivity CSV: {e}")
        errors.append("conn_csv")

    section("7. CC-threshold connectivity CSV")
    cc_path = os.path.join(raw, "subjBrainDataConnectivity_groupConsider_CCThresh_6.3.csv")
    try:
        cc_csv = pd.read_csv(cc_path, nrows=5)
        ok(f"Loadable; sample shape {cc_csv.shape}")
    except Exception as e:
        fail(f"Cannot load CC-thresh CSV: {e}")
        errors.append("cc_csv")

    # ------------------------------------------------------------------ #
    # 8. .mat files via scipy
    # ------------------------------------------------------------------ #
    section("8. MAT files (via scipy)")
    try:
        from scipy.io import loadmat
    except ImportError:
        warn("scipy not installed -- skipping .mat checks")
        loadmat = None

    mat_files = [
        ("dataOrg.mat (legacy processed)", os.path.join(raw, "dataOrg.mat")),
        ("connectivityOut.mat", os.path.join(raw, "connectivityOut.mat")),
        ("completeCookData.mat", os.path.join(raw, "completeCookData.mat")),
        ("normalizedData_KP143703.mat (sample)", os.path.join(raw, "normalizedData_KP143703.mat")),
        ("dataOrg.mat (dev copy)", os.path.join(base, "data", "dataOrg.mat")),
    ]
    if loadmat:
        for label, path in mat_files:
            try:
                mat = loadmat(path, squeeze_me=True)
                keys = [k for k in mat.keys() if not k.startswith("__")]
                ok(f"{label}: keys={keys}")
            except NotImplementedError:
                try:
                    import h5py
                    with h5py.File(path, "r") as f:
                        keys = list(f.keys())
                    ok(f"{label} (HDF5/v7.3): keys={keys}")
                except ImportError:
                    warn(f"{label}: MATLAB v7.3 file, needs h5py (pip install h5py)")
                except Exception as e2:
                    fail(f"{label} (HDF5 fallback): {e2}")
                    errors.append(label)
            except Exception as e:
                fail(f"{label}: {e}")
                errors.append(label)

    # ------------------------------------------------------------------ #
    # 9. Existing pickle cache
    # ------------------------------------------------------------------ #
    section("9. Existing pickle caches")
    pkl_files = [
        ("gmwm3D.pkl", os.path.join(base, "data", "gmwm3D.pkl")),
        ("reducedGMWM.pkl", os.path.join(base, "data", "reducedGMWM.pkl")),
    ]
    for label, path in pkl_files:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                ok(f"{label}: keys={list(data.keys())}")
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        ok(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                ok(f"{label}: type={type(data)}")
        except Exception as e:
            fail(f"{label}: {e}")
            errors.append(label)

    # ------------------------------------------------------------------ #
    # 10. Cross-check subject IDs
    # ------------------------------------------------------------------ #
    section("10. Subject ID cross-check")
    try:
        gmwm_set = set(site_ids_gmwm)
        demo_ids = set(demo_df["newids"].astype(str))
        legacy_ids = set(legacy_df["Site_ID"].astype(str))

        ok(f"GMWM subjects: {len(gmwm_set)}")
        ok(f"Demo Excel subjects: {len(demo_ids)}")
        ok(f"Legacy demo subjects: {len(legacy_ids)}")

        in_demo = gmwm_set & demo_ids
        ok(f"GMWM in Demo Excel: {len(in_demo)}/{len(gmwm_set)}")

        in_legacy = gmwm_set & legacy_ids
        ok(f"GMWM in Legacy demo: {len(in_legacy)}/{len(gmwm_set)}")

        extra_demo = demo_ids - gmwm_set
        if extra_demo:
            warn(f"Demo Excel has {len(extra_demo)} subjects not in GMWM (expected: total 395 vs 360)")

        missing_gmwm = gmwm_set - demo_ids
        if missing_gmwm:
            fail(f"GMWM has {len(missing_gmwm)} subjects NOT in Demo Excel!")
            errors.append("id_mismatch")
        else:
            ok("All GMWM subjects found in Demo Excel")
    except Exception as e:
        fail(f"Cross-check failed: {e}")
        errors.append("cross_check")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    section("SUMMARY")
    if errors:
        print(f"  Completed with {len(errors)} issue(s): {errors}")
    else:
        print("  All checks passed. Data is ready for processing.")

    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
