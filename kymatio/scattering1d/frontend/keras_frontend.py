from ...frontend.keras_frontend import ScatteringKeras
from ...scattering1d.frontend.base_frontend import (
    ScatteringBase1D,
    TimeFrequencyScatteringBase,
)

from kymatio.tensorflow import (
    Scattering1D as ScatteringTensorFlow1D,
    TimeFrequencyScattering as TimeFrequencyScatteringTensorflow,
)

from tensorflow.python.framework import tensor_shape


class ScatteringKeras1D(ScatteringKeras, ScatteringBase1D):
    def __init__(self, J, Q=1, T=None, max_order=2, oversampling=0, stride=None):
        ScatteringKeras.__init__(self)
        self.J = J
        self._Q = Q
        self._T = T
        self._max_order = max_order
        self._oversampling = oversampling
        self._stride = stride
        self.out_type = "array"
        self.backend = None

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-1:])
        self.S = ScatteringTensorFlow1D(
            J=self.J,
            shape=shape,
            Q=self._Q,
            T=self._T,
            max_order=self._max_order,
            oversampling=self._oversampling,
            stride=self._stride,
        )
        ScatteringKeras.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        nc = self.S.output_size()
        k0 = max(self.J - self._oversampling, 0)
        ln = self.S.ind_end[k0] - self.S.ind_start[k0]
        output_shape = [input_shape[0], nc, ln]
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        keys = ["J", "Q", "max_order", "oversampling"]
        return {key: getattr(self, key) for key in keys}


ScatteringKeras1D._document()


class TimeFrequencyScatteringKeras(ScatteringKeras, TimeFrequencyScatteringBase):
    def __init__(
        self,
        J,
        J_fr,
        Q,
        T=None,
        stride=None,
        Q_fr=1,
        F=None,
        stride_fr=None,
        format="joint"
    ):
        ScatteringKeras.__init__(self)
        self.J = J
        self.J_fr = J_fr
        self._Q = Q
        self._T = T
        self._stride = stride
        self._Q_fr = Q_fr
        self._F = F
        self._stride_fr = stride_fr
        self._format = format
        self.out_type = "array"
        self.backend = None

    def build(self, input_shape):
        shape = tuple(tensor_shape.TensorShape(input_shape).as_list()[-1:])
        self.S = TimeFrequencyScatteringTensorflow(
            J=self.J,
            J_fr=self.J_fr,
            shape=shape,
            Q=self._Q,
            T=self._T,
            stride=self._stride,
            Q_fr=self._Q_fr,
            F=self._F,
            stride_fr=self._stride_fr,
            out_type=self.out_type,
            format=self._format,
        )
        ScatteringKeras.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        breakpoint()
        if self.format == "time":
            input_shape = tensor_shape.TensorShape(input_shape).as_list()
            nc = self.S.output_size()
            k0 = max(self.J - self._oversampling, 0)
            ln = self.S.ind_end[k0] - self.S.ind_start[k0]
            output_shape = [input_shape[0], nc, ln]
            return tensor_shape.TensorShape(output_shape)
        elif self.format == "joint":
            pass

    def get_config(self):
        keys = ["J", "J_fr", "Q", "Q_fr", "T", "F", "stride", "stride_fr", "format"]
        return {key: getattr(self, key) for key in keys}


TimeFrequencyScatteringKeras._document()

__all__ = ["ScatteringKeras1D", "TimeFrequencyScatteringKeras"]
