
import warnings, sys; warnings.filterwarnings("ignore")
from cc3d.core.PyCoreSpecs import (PottsCore, CellTypePlugin, VolumePlugin, ContactPlugin,
                                   BlobInitializer, CenterOfMassPlugin)
seed, steps, dim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
potts = PottsCore(dim_x=dim, dim_y=dim, dim_z=1, steps=steps, fluctuation_amplitude=10.0,
                  neighbor_order=2, random_seed=seed)
ct = CellTypePlugin("Condensing", "NonCondensing")
vol = VolumePlugin()
vol.param_new("Condensing", target_volume=25, lambda_volume=2.0)
vol.param_new("NonCondensing", target_volume=25, lambda_volume=2.0)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Condensing", 16); con.param_new("Medium", "NonCondensing", 16)
con.param_new("Condensing", "Condensing", 2); con.param_new("NonCondensing", "NonCondensing", 11)
con.param_new("Condensing", "NonCondensing", 11)
com = CenterOfMassPlugin()
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 3, center=(dim // 2, dim // 2, 0),
                cell_types=("Condensing", "NonCondensing"))
body = "\n".join(s.xml.getCC3DXMLElementString() for s in (potts, ct, vol, con, com, blob))
sys.stdout.write('<CompuCell3D Revision="0" Version="4.10.0">\n' + body + '\n</CompuCell3D>\n')
