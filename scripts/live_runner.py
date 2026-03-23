import os, sys, pathlib
os.environ.setdefault('PYTHONIOENCODING','utf-8')
# Reconfigure stdout/stderr to UTF-8 before logging is set up.
# On Windows, sys.stdout uses the system codepage (cp1252) by default, which
# cannot encode Vietnamese characters.  PYTHONIOENCODING is set above but only
# takes effect for a fresh interpreter — if stdout was already opened we must
# reconfigure it explicitly.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

# Load .env manually
env_path = pathlib.Path('.env')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k,v = line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip())

import argparse as _ap, logging

# Parse --config early so we can use config-based log file name
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--config', default='configs/settings.yaml')
_pre_args, _ = _pre.parse_known_args()

# Derive log file from config name (e.g. live_acc2.yaml -> live_bot_log_acc2.txt)
_cfg_stem = pathlib.Path(_pre_args.config).stem  # e.g. 'live_ict_wyckoff' or 'live_acc2'
if 'acc2' in _cfg_stem:
    _log_file = 'outputs/live_bot_log_acc2.txt'
else:
    _log_file = 'outputs/live_bot_log.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # stdout already reconfigured to utf-8 above
        logging.FileHandler(_log_file, encoding='utf-8'),
    ]
)

from xauusd_ai.app import main
raise SystemExit(main())
