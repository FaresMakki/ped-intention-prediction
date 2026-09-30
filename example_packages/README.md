# Upload-ready PIE packages

These archives implement version 1 of the exact-model package contract and can
be uploaded directly from either website page.

| Package | Ground truth | PIE source |
| --- | --- | --- |
| `pie_no_crossing_example.zip` | No Crossing | set03 / video_0001 / pedestrian 3_1_220 |
| `pie_crossing_example.zip` | Crossing | set03 / video_0001 / pedestrian 3_1_267 |
| `pie_irrelevant_example.zip` | Irrelevant | set03 / video_0001 / pedestrian 3_1_236 |

Each ZIP contains exactly:

```text
manifest.json
tensors.npz
frames/frame_01.jpg
...
frames/frame_15.jpg
```

`tensors.npz` carries all 15 validated arrays needed by the final
Trajectory + Social pipeline and the five-method comparison page. Arrays use
the same normalization and schemas as the checkpoints.

Regenerate the examples from the prepared deployment data with:

```powershell
..\.venv\Scripts\python.exe scripts\create_example_packages.py
```

The API rejects unknown files, unsafe paths, malformed images, non-consecutive
frame metadata, incorrect tensor shapes or dtypes, non-finite values, and
packages larger than 25 MB.
