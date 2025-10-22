import re
import base64
from pathlib import Path

mdp = Path(r"d:\Coding Files\Netfotech\cric_agent\insights\1244025_insights.md")
text = mdp.read_text(encoding='utf-8')
# find first data URI
m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=\n]+)", text)
if not m:
    print('No data URI found')
    raise SystemExit(1)
enc = m.group(1)
# remove whitespace/newlines
enc = ''.join(enc.split())
img = base64.b64decode(enc)
out = Path(r"d:\Coding Files\Netfotech\cric_agent\insights\images\phase_rpo_per_team_1244025.png")
out.write_bytes(img)
# create linked md
linked = mdp.parent / (mdp.stem + '_linked.md')
new_text = text.replace(m.group(0), str(out.as_posix()).replace('\\','/'))
# wrap with image markdown
new_text = new_text.replace(str(out.as_posix()).replace('\\','/'), f"insights/images/{out.name}")
linked.write_text(new_text, encoding='utf-8')
print('Wrote', out, 'and', linked)
