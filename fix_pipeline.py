with open('run_full_pipeline.py', 'r') as f:
    code = f.read()

if 'lower_yellow' in code:
    print('Found color detection - removing...')
    start = code.find('# Calibrated Range')
    end = code.find('total_targets = len(std_preds)')
    if start != -1 and end != -1:
        code = code[:start] + code[end:]
    code = code.replace(
        'total_targets = len(std_preds) + len(machinery_boxes)',
        'total_targets = len(std_preds)'
    )
    code = code.replace(
        'tqdm.write(f"[{fname}] Detected {len(std_preds)} standard + {len(machinery_boxes)} construction = {total_targets} total")',
        'tqdm.write(f"[{fname}] Detected {total_targets} vehicles")'
    )
    start2 = code.find('for (mx1, my1, mx2, my2) in machinery_boxes:')
    end2 = code.find('# Blur mask edges')
    if start2 != -1 and end2 != -1:
        code = code[:start2] + code[end2:]
    with open('run_full_pipeline.py', 'w') as f:
        f.write(code)
    print('Done! Color detection removed.')
else:
    print('Already clean')