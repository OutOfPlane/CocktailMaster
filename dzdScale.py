import serial



class Scale:
    def __init__(self, port="/dev/ttyUSB0"):
        self.port = serial.Serial(port, timeout=2)
        self.value = None
        self.stable = False
        self.unit = ""

    def read_weight(self):
        self.port.write(b"\x1B\x70")
        data = self.port.read(14)
        if(data.endswith(b"\r\n") and len(data) == 14):
            # valid packet
            sign = data[0:2].decode("ascii").strip()
            value = float(sign + data[2:10].decode("ascii").strip())
            unit = data[10:12].decode("ascii").strip()
            self.value = value
            if(unit != ""):
                self.stable = True
                self.unit = unit # update unit
            else:
                self.stable = False
            return value, unit
        else:
            raise ValueError("Invalid data received from scale")
        
    def init(self):
        while(self.unit != "g"):
            while(self.unit == ""):
                self.read_weight()
            if(self.unit != "g"):
                # send cmd to change unit
                self.port.write(b"\x1B\x73")
                self.unit = "" # reset unit to trigger re-read



if(__name__ == "__main__"):
    scale = Scale("/dev/ttyUSB0")
    scale.init()
    while (1):
        value, unit = scale.read_weight()
        print(f"Value: {value} {unit}")