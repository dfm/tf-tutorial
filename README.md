This repository contains an interactive IPython worksheet (`worksheet.ipynb`)
designed to introduce you to model fitting using TensorFlow. Only very minimal
experience with Python should be necessary to get something out of this.

Prerequisites
-------------

You'll need the standard scientific Python stack (numpy, scipy, and
matplotlib), a recent version of [Jupyter](http://jupyter.org/), and
[emcee](http://dfm.io/emcee) installed. If you
don't already have a working Python installation (and maybe even if you do), I
recommend using the [Anaconda distribution](http://continuum.io/downloads) and
then running `pip install emcee`.


Usage
-----

After you have your Python environment set up, download the code from this
repository by running:

```
git clone https://github.com/dfm/tf-tutorial.git
```

or by [clicking here](https://github.com/dfm/tf-tutorial/archive/master.zip).

Then, navigate into the `tf-tutorial` directory and run

```
cp worksheet.ipynb worksheet_in_progress.ipynb
jupyter notebook
```

Then, in your web browser, navigate to [localhost:8888](http://localhost:8888)
and click on `worksheet_in_progress.ipynb` to get started.
