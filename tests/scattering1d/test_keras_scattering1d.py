import pytest
from tensorflow.keras.layers import Input, Flatten, Dense
from kymatio.keras import Scattering1D
from tensorflow.keras.models import Model
import os
import numpy as np
import io

def test_Scattering1D():
  """
  Applies scattering on a stored signal to make sure its output agrees with
  a previously calculated version.
  """
  test_data_dir = os.path.dirname(__file__)
  with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
      buffer = io.BytesIO(f.read())
      data = np.load(buffer)
  x = data['x']
  J = data['J']
  Q = data['Q']
  Sx0 = data['Sx']
  # default
  inputs0 = Input(shape=(x.shape[-1]))
  scat0 = Scattering1D(J=J, Q=Q)(inputs0)
  model0 = Model(inputs0, scat0)
  model0.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  Sg0 = model0.predict(x)
  assert np.allclose(Sg0, Sx0, atol=1e-07)
  # adjust T
  sigma_low_scale_factor = 2
  T=2**(J-sigma_low_scale_factor)
  inputs1 = Input(shape=(x.shape[-1]))
  scat1 = Scattering1D(J=J, Q=Q, T=T)(inputs1)
  model1 = Model(inputs1, scat1)
  model1.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  Sg1 = model1.predict(x)
  assert Sg1.shape == (Sg0.shape[0], Sg0.shape[1], Sg0.shape[2]*2**(sigma_low_scale_factor))
  