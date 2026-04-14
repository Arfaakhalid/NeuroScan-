"""
install_dependencies.py  --  NeuroScan Pro
"""
import subprocess, sys, importlib.util, platform

PACKAGES = [
    ("streamlit",    "streamlit>=1.28.0"),
    ("numpy",        "numpy>=1.24.0"),
    ("pandas",       "pandas>=2.0.0"),
    ("plotly",       "plotly>=5.17.0"),
    ("scipy",        "scipy>=1.11.0"),
    ("sklearn",      "scikit-learn>=1.3.0"),
    ("joblib",       "joblib>=1.3.0"),
    ("mne",          "mne>=1.4.0"),
    ("pyedflib",     "pyedflib>=0.1.28"),
    ("pywt",         "PyWavelets>=1.4.0"),
    ("antropy",      "antropy>=0.1.5"),
    ("PIL",          "Pillow>=10.0.0"),
    ("openpyxl",     "openpyxl>=3.1.0"),
    ("matplotlib",   "matplotlib>=3.7.0"),
]

def is_ok(imp_name):
    return importlib.util.find_spec(imp_name) is not None

def install():
    print("🧠 NeuroScan Pro — Installing dependencies")
    print("=" * 55)
    ok, fail = [], []
    for imp_name, pip_spec in PACKAGES:
        pkg_name = pip_spec.split(">=")[0]
        if is_ok(imp_name):
            print(f"  ✅ {pkg_name} already installed"); ok.append(pkg_name); continue
        print(f"  📦 Installing {pip_spec} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   pip_spec, "--quiet", "--no-cache-dir"],
                                  stderr=subprocess.DEVNULL)
            print(f"  ✅ {pkg_name} installed"); ok.append(pkg_name)
        except subprocess.CalledProcessError:
            print(f"  ❌ FAILED: {pkg_name}"); fail.append(pkg_name)
    print("=" * 55)
    print(f"✅ {len(ok)} packages ready")
    if fail:
        print(f"❌ Failed: {', '.join(fail)}")
        print("   Try: pip install <pkg> --break-system-packages")
    else:
        print("🚀 All ready — run:  streamlit run app.py")

if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} | {platform.system()}")
    ans = input("Install packages? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        install()
    else:
        print("Skipped. Run:  pip install -r requirements.txt")
