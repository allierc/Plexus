
from cc3d import CompuCellSetup
from sortSteppables import DumpSteppable
CompuCellSetup.register_steppable(steppable=DumpSteppable(frequency=1))
CompuCellSetup.run()
