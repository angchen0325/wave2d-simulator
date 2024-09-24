from abc import ABC, abstractmethod

import cupy
import cupy as cp
import cupyx.scipy.signal


class SceneObject(ABC):
    """
    Interface for simulation scene objects.
    A scene object is anything defining or modifying the simulation scene.
    For example: Light sources, absorbers or regions with specific refractive index.
    Scene objects can change the simulated field and draw their contribution to the wave speed field
    and dampening field each frame.
    """

    @abstractmethod
    def render(self, wave_speed_field: cupy.ndarray, dampening_field: cupy.ndarray):
        """
        Rendering the scene objects that contribute to the wave speed field and dampening field.
        """
        pass

    @abstractmethod
    def update_field(self, field: cupy.ndarray, t):
        """
        Performing updates to the field itself, e.g. for adding sources.
        """
        pass
