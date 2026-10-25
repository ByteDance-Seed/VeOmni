"""SeedOmni data helpers.

Importing this package registers ``data_type="seedomni"`` and its source
preprocessors. ``veomni.data`` does not import it, since it pulls in
``veomni.models.seed_omni``; the SeedOmni trainer does.
"""

from . import preprocess, seedomni_transform  # noqa: F401
