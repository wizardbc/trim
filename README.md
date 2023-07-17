# TRIM 
Trimmed Rank with Inverse softMax probability.

Universal and reliable, but simple, OOD(Out-of-Distribution) score.

## Datasets

### ImageNet-1k as ID
* [ILSVRC-2012/val](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) [[download](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)]
* run `valprep.sh` (from [soumith/imagenetloader.torch](https://github.com/soumith/imagenetloader.torch)) after untar the downloaded `ILSVRC2012_img_val.tar` file:
  ```bash
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```
### OOD for ImageNet-1k
From [deeplearning-wisc/dice](https://github.com/deeplearning-wisc/dice#out-of-distribution-dataset)
* [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) [[download](https://thor.robots.ox.ac.uk/datasets/dtd/dtd-r1.0.1.tar.gz)]
  ```bash
  rm -rf dtd/imdb dtd/labels
  ```
* [iNaturalist](https://github.com/visipedia/inat_comp/tree/master/2017) [[download](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz)]
* [SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html) [[download](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz)]
* [Places](http://places2.csail.mit.edu/) [[download](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz)]

From [hendrycks/natural-adv-examples](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-O](https://github.com/hendrycks/natural-adv-examples) [[download](https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar)]

From [ViM](https://github.com/haoqiwang/vim/tree/master)
* OpenImage-O: a subset of the [OpenImage-V3](https://github.com/openimages/dataset/blob/main/READMEV3.md) testing set. [[filelist](https://raw.githubusercontent.com/haoqiwang/vim/master/datalists/openimage_o.txt)]

