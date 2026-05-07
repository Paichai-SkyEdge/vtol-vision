# Basket + Mannequin Labeled Flat Dataset

One flat directory containing only labeled image/YOLO-label pairs.

Class order:

```txt
0 basket
1 mannequin
```

Each usable sample has:

```txt
<stem>.jpg
<stem>.txt
```

YOLO label format:

```txt
class_id center_x center_y width height
```

Coordinates are normalized to `[0, 1]`.
