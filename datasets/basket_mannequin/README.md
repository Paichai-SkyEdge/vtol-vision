# Basket + Mannequin YOLO Dataset

Class order:

```txt
0 basket
1 mannequin
```

Raw images are currently in `images/skyedge_vision`.

For labelImg:

```bash
tools/labelimg_basket_mannequin.sh
```

In labelImg, switch the save format to `YOLO`, then set the save directory to
`images/skyedge_vision` so each image gets a matching `.txt` label file.

The version-controlled copies of the class and dataset config are:

- `config/basket_mannequin_classes.txt`
- `config/basket_mannequin_dataset.yaml`
