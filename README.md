# Graph WaveNet traffic forecast viewer

An interactive page that shows what a trained [Graph WaveNet](https://arxiv.org/abs/1906.00121) predicts for highway
traffic in Los Angeles, next to the speeds that were actually measured. Built by Jiajie Su.

![Viewer screenshot](docs/screenshot.png)

The model forecasts speed at 207 loop-detector sensors (the METR-LA dataset) from the previous hour of readings.
In the page you can:

- pick one of 12 sensors and a forecast horizon of 15, 30 or 60 minutes;
- silence 0 to 50% of the sensors and watch the forecast degrade, using the same masking as the robustness test;
- read the error for the current view and for the whole test set at each masking level;
- see which sensors the learned adjacency matrix links most strongly to the selected one.

Nothing is computed in the browser. The predictions in `data/demo_data.json` were produced by running the model
once, and the page is a single static HTML file.

## Results behind the page

Whole test set (6,850 windows, all 12 forecast steps), mean absolute error in mph:

| Sensors silent | 0% | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|---|
| MAE | 3.04 | 3.57 | 4.23 | 5.08 | 6.04 | 7.01 |
| RMSE | 6.10 | 6.71 | 7.45 | 8.39 | 9.44 | 10.37 |

Masking zeroes the whole input history of a random share of sensors for each input window, with a fixed random seed.
Numbers move a little with the seed.

The sensors in the chart are the 12 with the most speed variation over the two days shown (Tuesday 2012-06-19 and
Wednesday 2012-06-20), so errors for the selected view are higher than the whole-test-set figures.

## Run it

Open `docs/index.html` in any browser, or serve the folder:

```
python3 -m http.server 8765 --directory docs
```

The page loads two fonts from Google Fonts. Without a network connection it falls back to system fonts.

## Rebuild or re-export

Build the page from the source and data (standard library only):

```
python3 scripts/build.py
```

Regenerate the data from the model. This needs PyTorch and the original Graph WaveNet code and data:

```
git clone https://github.com/nnzhan/Graph-WaveNet ../Graph-WaveNet   # then prepare METR-LA as its README describes
pip install -r requirements-export.txt
python scripts/export_data.py            # fast
python scripts/export_data.py --full     # also recompute the whole-test-set numbers, about 15 minutes on CPU
```

Set `GWNET_REPO` or `GWNET_CKPT` to use a different clone or checkpoint.

## Layout

```
docs/index.html            built page, served by GitHub Pages from /docs
src/template.html          page source (HTML, CSS, JavaScript)
data/demo_data.json        exported predictions for the 12 sensors, two days, three horizons, six masking levels
data/full_test_summary.json   whole-test-set MAE and RMSE per masking level
scripts/build.py           inlines the data into the template
scripts/export_data.py     runs the checkpoint and writes the data files
checkpoint/                trained weights used for the export (1.2 MB)
```

## Credits and data

- Model: Z. Wu, S. Pan, G. Long, J. Jiang, C. Zhang, *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI 2019.
  The model code used to export the data is the authors' PyTorch implementation, released under the MIT license at
  https://github.com/nnzhan/Graph-WaveNet. This project does not include that code.
- Data: METR-LA, 5-minute speed readings from 207 loop detectors on Los Angeles County highways, as prepared by
  Li et al., *Diffusion Convolutional Recurrent Neural Network*, ICLR 2018. The raw data is not included here.
- The checkpoint was trained for a course project (UCI EECS 298E, Winter 2026).

## Not yet decided

No license has been chosen for this repository's own code.
