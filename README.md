# Constrained LDA for Strain Inference

## Install and Test

To install run `python setup.py develop`

To test run `python -m pytest`

## How to Use

```
cd constrained_lda/
python cli.py constrained_lda ../tests/test.cfg --debug > ../tests/test.out 2>&1 &
```

## Directory Structure

**constrained_lda/**: CLI and API containing all constrained LDA functions.

**tests/**: Toy datasets for code validation.

**latex_docs/**: Project proposal, milestone, and final reports.

**hoffman_deprecated/**: Application of standard LDA to infer topics from Wikipedia. From the original authors (Hoffman et al.).

## Changelog

### Basic Implementation

11/15/2020: Functional version of basic LDA adapted from Hoffman et al. with toy dataset validation. 

### Setup

11/2/2020: Basic project structure established but nothing is functional yet (i.e.: can't be installed with `setup.py`, etc.).

---

# From M. Hoffman's Implementation

ONLINE VARIATIONAL BAYES FOR LATENT DIRICHLET ALLOCATION

Matthew D. Hoffman
mdhoffma@cs.princeton.edu

(C) Copyright 2010, Matthew D. Hoffman

This is free software, you can redistribute it and/or modify it under
the terms of the GNU General Public License.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

This software is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA

------------------------------------------------------------------------

This Python code implements the online Variational Bayes (VB)
algorithm presented in the paper "Online Learning for Latent Dirichlet
Allocation" by Matthew D. Hoffman, David M. Blei, and Francis Bach,
to be presented at NIPS 2010.

The algorithm uses stochastic optimization to maximize the variational
objective function for the Latent Dirichlet Allocation (LDA) topic model.
It only looks at a subset of the total corpus of documents each
iteration, and thereby is able to find a locally optimal setting of
the variational posterior over the topics more quickly than a batch
VB algorithm could for large corpora.


Files provided:
* onlineldavb.py: A package of functions for fitting LDA using stochastic
    optimization.
* onlinewikipedia.py: An example Python script that uses the functions in
    onlineldavb.py to fit a set of topics to the documents in Wikipedia.
* wikirandom.py: A package of functions for downloading randomly chosen
    Wikipedia articles.
* printtopics.py: A Python script that displays the topics fit using the
    functions in onlineldavb.py.
* dictnostops.txt: A vocabulary of English words with the stop words removed.
* readme.txt: This file.
* COPYING: A copy of the GNU public license version 3.

You will need to have the numpy and scipy packages installed somewhere
that Python can find them to use these scripts.


Example:
python onlinewikipedia.py 101
python printtopics.py dictnostops.txt lambda-100.dat

This would run the algorithm for 101 iterations, and display the
(expected value under the variational posterior of the) topics fit by
the algorithm. (Note that the algorithm will not have fully converged
after 101 iterations---this is just to give an idea of how to use the
code.)
