import os, sys, pathlib
os.environ.setdefault('PYTHONIOENCODING','utf-8')
sys.path.insert(0, 'src')

# Load .env manually
env_path = pathlib.Path('.env')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k,v = line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip())

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/live_bot_log.txt', encoding='utf-8'),
    ]
)

from xauusd_ai.app import main
raise SystemExit(main())
