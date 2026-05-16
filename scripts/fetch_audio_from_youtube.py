from yt_dlp import YoutubeDL  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from paths import data_path


def get_playlist_urls(playlist_url):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        return [entry["url"] for entry in info["entries"] if entry is not None]


def download_single(url, output_dir):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(playlist_index)03d - %(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_playlist_parallel(playlist_url, output_dir, workers=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = get_playlist_urls(playlist_url)

    print(f"Found {len(urls)} videos")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_single, url, output_dir) for url in urls]

        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
                print(f"[{i}/{len(urls)}] done")
            except Exception as e:
                print(f"Failed: {e}")


if __name__ == "__main__":
    url = "https://www.youtube.com/playlist?list=PLUl4u3cNGP60cspQn3N9dYRPiyVWDd80G"
    download_playlist_parallel(url, output_dir=data_path("lectures_audio"), workers=10)
