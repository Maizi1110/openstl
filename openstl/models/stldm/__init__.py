from openstl.models.stldm.stldm import model_setup
from openstl.models.stldm.stldm_spatial import model_setup as spatial_setup
from openstl.models.stldm.inference import InferenceHub


n2n_setup = {'2D': spatial_setup, '3D': model_setup}