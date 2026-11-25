# Windows Helper Commands

These are example PowerShell commands that proved useful while running, checking, and troubleshooting the local PDF index on Windows.

Replace placeholder paths like `<repo-root>` and `<your-pdf-root>` with your own local paths.

## Check Current Active Indexed Count

```powershell
@'
import csv, pickle, zlib
from pathlib import Path

repo = Path(r'<repo-root>')
manifest_path = repo / 'manifest.csv'
active_files_zip = repo / 'index' / '<active-index-name>' / 'files.zip'

with open(manifest_path, newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

manifest_set = {row['file_location'].replace('/', '\\') for row in rows if row.get('file_location')}
data = pickle.loads(zlib.decompress(active_files_zip.read_bytes()))

indexed = set()
for file_location, status in data.items():
    if status == 'ERROR':
        continue
    normalized = str(file_location).replace('/', '\\')
    if normalized in manifest_set:
        indexed.add(normalized)

print(len(indexed))
'@ | python -
```

Use the active shard folder name shown in the app under **Active index details**, printed by your most recent rebuild, or visible under `<repo-root>\index`, for example `pqa_index_cc6c1744115a46e85b221f1f29f61f2d`.

## Check Whether Rebuild Is Still Running

A rebuild started with `pdf-research-rebuild` shows that name; one started from the app runs `rebuild.py`.

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'pdf-research-rebuild|rebuild\.py' } | Select-Object ProcessId, Name, CommandLine
```

## Follow A Rebuild Started From The App

```powershell
Get-Content <repo-root>\index\rebuild.log -Tail 20 -Wait
```

## Check Recent Index File Activity

```powershell
Get-ChildItem -Recurse <repo-root>\index | Sort-Object LastWriteTime -Descending | Select-Object -First 10 FullName, LastWriteTime
```

## Find Missing PDFs From The Manifest

```powershell
@'
import csv, pickle, zlib
from pathlib import Path

repo = Path(r'<repo-root>')
manifest_path = repo / 'manifest.csv'
active_files_zip = repo / 'index' / '<active-index-name>' / 'files.zip'

with open(manifest_path, newline='', encoding='utf-8-sig') as f:
    manifest_rows = list(csv.DictReader(f))

manifest_files = {
    row['file_location'].replace('/', '\\')
    for row in manifest_rows
    if row.get('file_location')
}

data = pickle.loads(zlib.decompress(active_files_zip.read_bytes()))
indexed_ok = {
    str(file_location).replace('/', '\\')
    for file_location, status in data.items()
    if status != 'ERROR'
}

missing = sorted(manifest_files - indexed_ok)
print('Missing:', len(missing))
for m in missing:
    print(m)
'@ | python -
```

## Find Streamlit And Python Processes

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|streamlit' } | Select-Object ProcessId, Name, CommandLine
```

## Stop A Stuck Process

```powershell
Stop-Process -Id <PID> -Force
```

## Rebuild The Index Explicitly

```powershell
cd <repo-root>
pdf-research-rebuild
```

## Run The Streamlit UI

```powershell
cd <repo-root>
streamlit run src/pdf_research_assistant/app.py
```

## What Helped Most

- Check the terminal output, not just the Streamlit UI
- Confirm whether `pdf-research-rebuild` is still running
- Check the indexed count on disk
- Compare `manifest.csv` against the indexed files
- Look for duplicate or stuck Streamlit/Python processes
- Use `pdf-research-rebuild` for a controlled rebuild instead of relying only on first-query indexing
