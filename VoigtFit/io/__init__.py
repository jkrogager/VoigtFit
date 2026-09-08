from ..fits_input import load_fits_spectrum
from .. import fits_input hdf5_save output parse_input
import warnings
warnings.warn(
	"VoigtFit.io is deprecated; import from VoigtFit directly instead.",
	DeprecationWarning,
	stacklevel=2,
)
