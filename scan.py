"""python scan.py  -> index samples/ into library.db and print counts."""
from mcp_daw import library

if __name__ == "__main__":
    print("indexed", library.scan("samples", "library.db"))
    for cat, kind, n in library.counts("library.db"):
        print(f"{cat:<10} {kind:<9} {n}")
