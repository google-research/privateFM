# Private FM Sketch
Code for differentially private Flajolet-Martin sketch.

This is not an officially supported Google product.


## Install dependencies
```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

## Running the simulation
`k` is the true cardinality of the synthetic data, `epsilon` is the desired DP level, `m` is the number of repeatation, and `gamma` is the accuracy parameter in the algorithm.
```bash
python expt_simulate.py --k=4096 --epsilon=1.0 --m=1024 --gamma=1.0
```

## Example simulation to run
To examine the results of the algorithm, you can try a few values for the true cardinality `k`, `epsilon`, `m` and `gamma`. 
```bash
for k in 4096 8192 16384 32768 65536 131072 262144 524288 1048576; do
  for epsilon in -1.0 1.0; do
    for m in 1024 4096 32768; do
      for gamma in 1.0 0.01; do
        python expt_simulate.py --k=$k --epsilon=${epsilon} --m=$m --gamma=${gamma} 
      done
    done
  done
done
```

Then you can plot the MRE (mean relative error) with the following command. Some parameters are hard-coded in `plot_result.py`. Please change them accordingly.
```bash
for epsilon in -1.0 1.0; do
  for gamma in 1.0 0.01; do
    python plot_result.py --epsilon=${epsilon} --gamma=${gamma} 
  done
done
```