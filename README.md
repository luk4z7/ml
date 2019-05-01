# Machine Learning

## Getting-started

## Install

### prerequisites
- python 2
  - numpy
  - pandas
  - sklearn
  - matplotlib
  - ipython
  - scikit-image
  - tensorflow 
  - keras
  - h5py
  - skimage
  - jupyter
  - Pillow

clone the project
```bash
git clone https://github.com/luk4z7/ml
```

Install dependencies
```bash
cd ml

pip install -r requirements.txt --user
```

Execute notebook jupyter
```bash
jupyter notebook --allow-root
```

## docker

```bash
docker pull jupyter/datascience-notebook
```

```bash
cd ml

docker run -d --rm -p  10000:8888 -v "$PWD":/home/jovyan/work jupyter/datascience-notebook
```

