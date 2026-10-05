# Prepared video examples

Each numbered folder contains the two files required by the website's **Video
upload** page:

```text
clip.mp4
clip_data.zip
```

The MP4 is an annotated, presentation-speed encoding of exactly 15 consecutive
PIE frames. The data ZIP contains the corresponding version-2 tensors and all
metadata needed by every deployed comparison branch. Its manifest stores the
MP4 SHA-256 digest, so files from different folders cannot be mixed.

| Folder | Ground truth | PIE source window | Pedestrians |
| --- | --- | --- | --- |
| `01_no_crossing_4ped` | No Crossing | set03 / video_0010 / 3749–3763 | 3_10_644, 3_10_645, 3_10_646, 3_10_647 |
| `02_crossing_4ped` | Crossing | set03 / video_0012 / 7432–7446 | 3_12_746, 3_12_747, 3_12_749, 3_12_750 |
| `03_crossing_4ped` | Crossing | set03 / video_0015 / 3158–3172 | 3_15_827, 3_15_828, 3_15_829, 3_15_831 |
| `04_irrelevant_4ped` | Irrelevant | set04 / video_0003 / 2348–2362 | 4_3_1129, 4_3_1130, 4_3_1131, 4_3_1132 |
| `05_no_crossing_4ped` | No Crossing | set04 / video_0004 / 13040–13054 | 4_4_1195, 4_4_1196, 4_4_1197, 4_4_1198 |

The source frames were recorded at 30 FPS. The exported MP4s play at 10 FPS so
the short observation window remains visible during a presentation; the model
still receives the unchanged tensors for the original consecutive frames.

Regenerate all ZIP and prepared-video examples from the deployment root:

```powershell
..\.venv\Scripts\python.exe scripts\create_example_packages.py
```
