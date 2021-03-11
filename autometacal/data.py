import numpy as np
import random
import galsim
from astropy.table import Table
import tensorflow_datasets as tfds

from . import galaxies

_DESCRIPTION = """Automatatic meta calibration."""
_URL = 'https://github.com/CosmoStat/autometacal'
_CITATION = r"""@article{my-awesome-dataset-2021, author = {Cosmostat},}"""

class GalGen(tfds.core.GeneratorBasedBuilder):
  """Galaxy Generator for Tensorflow Operations."""

  VERSION = tfds.core.Version('0.0.0')
  RELEASE_NOTES = {'0.0.0': "Initial code."}


  def __init__(self,
               data_file: str,
               drawer,
               train_split=.75,
               stamp_size=50,
               channels=1,
               galsim_drawing_method = "no_pixel",
               galsim_interpolator = "linear"):
        """
        Inputs:
            data_file: table from where shape parameters, size and snr are read
            drawer: a galaxy drawer method from galaxies.py
            stamp_size : the size of the square images to be generated
            channels: how many channels (different filters) for each galaxy
            galsim_drawing_method: see galsim
            galsim_interpolator: not yet implemented
        """
    self.data = Table.read(data_file)
    self.stamp_size = stamp_size
    self.channels = channels
    self.train_split = train_split
    self.drawer = lambda galaxy,psf : drawer(galaxy,
                                             psf,
                                             stamp_size = self.stamp_size,
                                             channels = self.channels,
                                             method = galsim_drawing_method,
                                             interpolator = galsim_interpolator)
    super(GalGen,self).__init__()

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict({
          'image_description': tfds.features.Text(),
          'image': tfds.features.Image(shape=(self.stamp_size ,
                                              self.stamp_size ,
                                              self.channels)),
          'label': tfds.features.ClassLabel(num_classes=2),
          }),
      supervised_keys=('image', 'label'),
      citation=_CITATION)

  def _split_generators(self):
    """Returns generators according to split."""

    intsplit = int(np.round(len(self.data)*self.train_split))
    train_slice = np.random.choice(np.arange(len(self.data)),intsplit,replace=False)
    test_slice = [i for i in np.arange(len(self.data))]
    _ = [test_slice.remove(train_selected) for train_selected in train_slice]
    return {'train': self._generate_examples(self.data[train_slice]),
             'test': self._generate_examples(self.data[test_slice]),
             }

  def _generate_examples(self, data):
      """Yields examples."""
    for i, galaxy in enumerate(data):
      image = self.drawer(galaxy,None)
      g1, g2 = galaxy['g1'], galaxy['g2']
      yield i, g1, g2, image


class GalGenCOSMOS(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for COSMOS GalSim dataset."""
  def __init__(self):
      print("NOT IMPLEMENTED! COMING SOON")
  VERSION = tfds.core.Version('0.0.0')
  RELEASE_NOTES = {
      '0.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract('https://zenodo.org/record/3242143/files/COSMOS_25.2_training_sample.tar.gz')
    cat = galsim.COSMOSCalog('COSMOS_25.2_training_sample')
    split = 0.7
    trainlist = [0:round(split*len(cat))]
    testlist = [0:round(split*len(cat))]
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(cat,trainlist),
        'test': self._generate_examples(cat,testlist),
    }

  def _generate_examples(self, cat, sublist) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for i in sublist:
      gal = cat.makeGalaxy(i)
      image =gal.original_gal.image.array

      yield i ,image

class GalGenHSC(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for HSC simulated galaxies GalSim dataset."""
  def __init__(self):
      print("NOT IMPLEMENTED! COMING SOON")

  VERSION = tfds.core.Version('0.0.0')
  RELEASE_NOTES = {
      '0.0.0': 'Initial release.',
  }

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description="HSC Weak Lensing Image Simulations",
      homepage="https://hsc-release.mtk.nao.ac.jp/doc/index.php/weak-lensing-simulation-catalog-pdr1/",
      features=tfds.features.FeaturesDict({
          'image_description': tfds.features.Text(),
          'image': tfds.features.Image(shape=(stamp_size ,
                                              stamp_size ,
                                              1)),
          }),
      supervised_keys=('image'),
      citation="Mandelbaum 2018, arXiv:1710.00885")


  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    data_choices = ["parent_best_processed", "parent_median_processed", "parent_worst_processed"]
    

    extracted_path = dl_manager.download_and_extract(f"https://hsc-release.mtk.nao.ac.jp/archive/pdr1_incremental/{data_choices[0]}.tar.gz")
    cat = galsim.RealGalaxyCatalog(data_choices[0]) ##?
    split = 0.7
    trainlist = [0:round(split*len(cat))]
    testlist = [0:round(split*len(cat))]
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(cat,trainlist),
        'test': self._generate_examples(cat,testlist),
    }

  def _generate_examples(self, cat, sublist) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for i in sublist:
      gal = cat.makeGalaxy(i)
      image =gal.original_gal.image.array

      yield i ,image
