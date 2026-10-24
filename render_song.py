"""python render_song.py songs/demo.json out/demo.wav [--mp3]"""
import shutil, subprocess, sys
from mcp_daw import render, song

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    s = song.load(src)
    print(song.describe(s))
    _, warns = render.render(s, "samples", dst, db="library.db")
    print("wrote", dst, "| licence warnings:", warns or "none")
    if "--mp3" in sys.argv and shutil.which("ffmpeg"):
        mp3 = dst.rsplit(".", 1)[0] + ".mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", dst, "-b:a", "192k", mp3], check=True)
        print("wrote", mp3)
