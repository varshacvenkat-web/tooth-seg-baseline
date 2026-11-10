## predict.py demo

Inputs: 10 test-split images (never trained on or used for checkpoint selection). Hard cases: 9.jpg.

```
121.jpg: 26 teeth -> [1, 2, 5, 6, 10, 12, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  (tooth numbers predicted more than once (check for duplicates): 12 x2)
171.jpg: 24 teeth -> [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
227.jpg: 32 teeth -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
230.jpg: 26 teeth -> [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31]
246.jpg: 16 teeth -> [2, 4, 11, 15, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31]
324.jpg: 29 teeth -> [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32]
486.jpg: 30 teeth -> [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
61.jpg: 30 teeth -> [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
73.jpg: 32 teeth -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
9.jpg: 18 teeth -> [2, 7, 8, 9, 17, 18, 20, 21, 22, 24, 24, 25, 26, 27, 28, 29, 30, 31]  (tooth numbers predicted more than once (check for duplicates): 24 x2)
Saved to /content/repo/results/demo/predictions  (10 ok, 0 failed)
```

### Teeth per image at different confidence thresholds

| Confidence threshold | Mean teeth per image | Min | Max |
| --- | --- | --- | --- |
| 0.1 | 29.7 | 18 | 37 |
| 0.25 | 28.8 | 17 | 36 |
| 0.5 | 26.3 | 16 | 32 |
| 0.75 | 22.3 | 8 | 31 |

### Edge cases

```
blank_no_teeth.png: 0 teeth -> []  (no teeth found above the confidence threshold)
colour_tinted.png: 18 teeth -> [5, 8, 17, 18, 20, 21, 22, 23, 24, 24, 25, 26, 26, 27, 28, 29, 30, 31]  (colour image converted to grayscale; tooth numbers predicted more than once (check for duplicates): 24 x2, 26 x2)
ERROR corrupted.jpg: Could not decode corrupted.jpg: the file is corrupted or not an image format OpenCV reads
rgba.png: 18 teeth -> [2, 7, 8, 9, 17, 18, 20, 21, 22, 24, 24, 25, 26, 27, 28, 29, 30, 31]  (alpha channel dropped; tooth numbers predicted more than once (check for duplicates): 24 x2)
sixteen_bit.png: 18 teeth -> [2, 7, 8, 9, 17, 18, 20, 21, 22, 24, 24, 25, 26, 27, 28, 29, 30, 31]  (16-bit image converted to 8-bit; tooth numbers predicted more than once (check for duplicates): 24 x2)
ERROR tiny_20px.png: tiny_20px.png: image is 20x20 px; at least 32 px per side is required
Saved to /content/repo/results/demo/edge_case_outputs  (4 ok, 2 failed)
Input not found: results/demo/edge_cases/does_not_exist.png
```

Outputs per image: JSON (tooth number, confidence, box, polygon), label PNG and N x H x W binary masks at the original resolution, overlay JPG drawn from the polygons (alignment check).
