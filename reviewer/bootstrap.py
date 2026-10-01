'Initialize a random private signing key without printing it.'
import os
from pathlib import Path
import secrets

path = Path(os.environ['REVIEW_KEY_FILE'])
path.parent.mkdir(parents=True, exist_ok=True)
try:
    with path.open('x') as out:
        out.write(secrets.token_hex(32))
except FileExistsError:
    pass
os.chown(path, 10001, 10001)
path.chmod(0o400)
print('Service identity initialized.')
