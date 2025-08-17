from importlib.metadata import version

pkgs = [
    "thop",
    "torch",
]
for p in pkgs:
    print(f"{p} version: {version(p)}")