import sys

path = r'c:\Users\Administrator\Documents\Trade-Indicator\src\xauusd_ai\orchestrator.py'

with open(path, 'rb') as f:
    content = f.read()

# Check if already patched
if b'_check_exit_model(latest_row, frames)' in content:
    print('Already patched: _check_exit_model(latest_row, frames) found')
else:
    # Insert _check_exit_model before DCA if block
    # Pattern: \r\n                if settings.execution.dca.enabled and settings.execution.auto_trade
    old = b'\r\n                if settings.execution.dca.enabled and settings.execution.auto_trade'
    new = (b'\r\n'
           b'                # Exit Model: check whether open positions should exit early\r\n'
           b'                _check_exit_model(latest_row, frames)\r\n\r\n'
           b'                if settings.execution.dca.enabled and settings.execution.auto_trade')

    count = content.count(old)
    print(f'Found {count} occurrence(s) of DCA if pattern')
    if count == 1:
        content = content.replace(old, new, 1)
        with open(path, 'wb') as f:
            f.write(content)
        print('SUCCESS: inserted _check_exit_model call before DCA block')
    else:
        print('ERROR: unexpected number of matches')
        sys.exit(1)

# Also insert in the chunked sleep loop
with open(path, 'rb') as f:
    content = f.read()

if b'_check_exit_model(_last_row_ctx, _last_frames_ctx)' in content:
    print('Already patched: chunked sleep exit model call found')
else:
    old2 = b'_check_closed_positions(_last_row_ctx, _last_frames_ctx)\r\n'
    new2 = (b'_check_closed_positions(_last_row_ctx, _last_frames_ctx)\r\n'
            b'                    _check_exit_model(_last_row_ctx, _last_frames_ctx)\r\n')

    count = content.count(old2)
    print(f'Found {count} occurrence(s) of chunked-sleep check pattern')

    if count >= 1:
        content = content.replace(old2, new2, 1)
        with open(path, 'wb') as f:
            f.write(content)
        print('SUCCESS: inserted _check_exit_model call in chunked sleep loop')
    else:
        print('WARNING: chunked sleep pattern not found - skipping')
