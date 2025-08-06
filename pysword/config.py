device = "cpu"

def set_device(deviceType):
    global device
    if device in ["cpu", "gpu"]:
        device = deviceType 
    else:
        raise ValueError("expected device type to be of cpu or gpu")

def get_device():
    return device

