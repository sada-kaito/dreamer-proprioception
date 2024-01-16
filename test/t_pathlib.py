import pathlib

current_dir = pathlib.Path('.')
print(current_dir.resolve())
current_dir.mkdir(parents=True, exist_ok=True)

datadir = current_dir / 'episodes'
print(datadir)
print(current_dir)
print(datadir.resolve())
datadir.mkdir(parents=True, exist_ok=True)
print(datadir.resolve())