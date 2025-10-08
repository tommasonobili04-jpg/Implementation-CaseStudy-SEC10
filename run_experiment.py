from pathlib import Path
import yaml

def main(cfg_path="configs/example.yaml"):
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        print(f"[WARN] Config {cfg_file} non trovata. Uso default.")
        config = {"seed": 42, "results_dir": "results"}
    else:
        config = yaml.safe_load(cfg_file.read_text())

    # crea cartella risultati
    out = Path(config.get("results_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)

    # TODO: qui chiamerai le funzioni reali (marginals, BL, solver duale, ecc.)
    print("[OK] Esperimento eseguito (placeholder). Output dir:", out.resolve())

if __name__ == "__main__":
    main()
