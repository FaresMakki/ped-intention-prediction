# Upload-ready PIE packages

The three main archives implement version 2 of the exact-model package contract
and can be uploaded directly from either website page. Each contains multiple
pedestrians aligned to the same 15 source frames.

| Package | Scene ground truth | PIE source | Pedestrians |
| --- | --- | --- | --- |
| `pie_no_crossing_example.zip` | No Crossing | set03 / video_0001 | 3_1_221, 3_1_222 |
| `pie_crossing_example.zip` | Crossing | set03 / video_0003 | 3_3_328, 3_3_329 |
| `pie_irrelevant_example.zip` | Irrelevant | set03 / video_0009 | 3_9_585, 3_9_586 |
| `pie_legacy_v1_single_person.zip` | No Crossing | set03 / video_0001 | 3_1_221 |

The following additional version-2 packages come from PIE windows that are not
used by any of the 15 built-in website scenes:

| Package | Scene ground truth | PIE source window | Pedestrians |
| --- | --- | --- | --- |
| `pie_extra_no_crossing_4ped.zip` | No Crossing | set03 / video_0010 / 3749–3763 | 3_10_644, 3_10_645, 3_10_646, 3_10_647 |
| `pie_extra_crossing_4ped_01.zip` | Crossing | set03 / video_0012 / 7432–7446 | 3_12_746, 3_12_747, 3_12_749, 3_12_750 |
| `pie_extra_crossing_4ped_02.zip` | Crossing | set03 / video_0015 / 3158–3172 | 3_15_827, 3_15_828, 3_15_829, 3_15_831 |

Each ZIP contains exactly:

```text
manifest.json
tensors.npz
frames/frame_01.jpg
...
frames/frame_15.jpg
```

In version 2, every array in `tensors.npz` starts with a pedestrian dimension.
The remaining dimensions are identical to the checkpoint inputs. The manifest
lists each pedestrian and provides its box (or `null`) on every frame. The
legacy archive demonstrates that the original unbatched version-1 format still
works.

Regenerate the built-in examples and the separate source-dataset examples with:

```powershell
..\.venv\Scripts\python.exe scripts\create_example_packages.py
```

The API rejects unknown files, unsafe paths, malformed images, non-consecutive
frame metadata, incorrect tensor shapes or dtypes, non-finite values, and
packages larger than 25 MB.
