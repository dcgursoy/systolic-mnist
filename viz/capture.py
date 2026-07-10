"""Capture README screenshots + an animated GIF from the visualizer.

Renders viz/index.html (file://, no server) in headless Chromium, seeks the
cycle scrubber, and saves:
  docs/img/viz_results.png   final frame: prediction + verdict panel
  docs/img/viz_wave.png      mid-inference systolic wave
  docs/img/viz_demo.gif      final-layer animation ending on the result

Usage: python viz/capture.py
"""

import io
import pathlib

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
URL = (ROOT / "viz" / "index.html").as_uri()


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1360, "height": 860})
        page.add_init_script("localStorage.setItem('sysarr_intro_seen','1')")
        page.goto(URL)
        page.wait_for_function(
            "window.PLAYER && window.PLAYER.frames.length > 0")

        def seek(i):
            page.evaluate(
                "(i) => { const s = document.getElementById('scrub');"
                "s.value = i; s.dispatchEvent(new Event('input')); }", i)

        nf = int(page.evaluate("window.PLAYER.frames.length"))
        l2 = int(page.evaluate(
            "window.PLAYER.frames.findIndex(f => f.l === 2)"))

        # hero: final results
        seek(nf - 1)
        page.screenshot(path=str(IMG / "viz_results.png"))

        # wave: first drain-phase cycle of layer 2 with the array mid-flight
        wave = int(page.evaluate(
            "window.PLAYER.frames.findIndex((f,i) => i > %d && f.st === 4 && f.dv)" % l2))
        seek(wave)
        page.screenshot(path=str(IMG / "viz_wave.png"))

        # GIF: run the whole final layer, ending held on the result
        frames = []
        for i in range(l2, nf, 2):
            seek(i)
            frames.append(Image.open(io.BytesIO(page.screenshot()))
                          .convert("P", palette=Image.ADAPTIVE, colors=128))
        durations = [80] * len(frames)
        durations[-1] = 3000  # hold the result
        frames[0].save(IMG / "viz_demo.gif", save_all=True,
                       append_images=frames[1:], duration=durations,
                       loop=0, optimize=True)
        browser.close()

    for f in ("viz_results.png", "viz_wave.png", "viz_demo.gif"):
        print(f, f"{(IMG / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
