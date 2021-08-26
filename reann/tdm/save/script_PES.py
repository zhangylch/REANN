import pes.PES as PES
import torch
def jit_pes():
    init_pes=PES.PES()
    checkpoint = torch.load("EANN.pth",map_location='cpu')
    init_pes.load_state_dict(checkpoint['eannparam'])
    scripted_pes=torch.jit.script(init_pes)
    for params in scripted_pes.parameters():
        params.requires_grad=False
    scripted_pes.save("EANN_PES_DOUBLE.pt")
    scripted_pes.to(torch.float32)
    scripted_pes.save("EANN_PES_FLOAT.pt")
