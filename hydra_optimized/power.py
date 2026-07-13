import pandas as pd

CSV_FILE = "power_profile_xilinx_u280_gen3x16_xdma_base_1-0.csv"

df = pd.read_csv(
    CSV_FILE,
    engine="python",
    skipinitialspace=True,
    skiprows=1
)

# Drop empty columns from trailing commas
# df = df.dropna(axis=1, how="all")

print("Columns:", df.columns.tolist())

# ---- Handle timestamp safely ----
time_col = None
for c in df.columns:
    if "time" in c.lower():
        time_col = c
        break

if time_col:
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.rename(columns={time_col: "timestamp"})

# ---- Power rails ----
rails = {
    "12v_aux": ("12v_aux_curr", "12v_aux_vol"),
    "12v_pex": ("12v_pex_curr", "12v_pex_vol"),
    "vccint":  ("vccint_curr",  "vccint_vol"),
    "3v3_pex": ("3v3_pex_curr", "3v3_pex_vol"),
}

for name, (i_col, v_col) in rails.items():
    if i_col in df.columns and v_col in df.columns:
        df[f"{name}_power_W"] = (df[i_col] / 1000.0) * (df[v_col] / 1000.0)

power_cols = [c for c in df.columns if c.endswith("_power_W")]
df["total_power_W"] = df[power_cols].sum(axis=1)

# ---- Stats ----
print("\n=== POWER STATISTICS (W) ===")
for c in power_cols:
    p = df[c]
    print(f"{c:18s} mean={p.mean():6.2f}  min={p.min():6.2f}  max={p.max():6.2f}")

tp = df["total_power_W"]
print("\nTOTAL BOARD POWER")
print(f"Mean : {tp.mean():.2f} W")
print(f"Min  : {tp.min():.2f} W")
print(f"Max  : {tp.max():.2f} W")
print(f"Std  : {tp.std():.2f} W")

# ---- Temperature stats ----
temp_cols = [c for c in df.columns if c.endswith("_temp")]

print("\n=== TEMPERATURE STATISTICS (°C) ===")
for c in temp_cols:
    t = df[c]
    print(f"{c:15s} mean={t.mean():5.1f}  max={t.max():5.1f}")
