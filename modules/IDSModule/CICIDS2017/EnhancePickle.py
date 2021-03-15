import pickle
from _io import BytesIO
from typing import Optional


class PickleWriteBuffer(BytesIO):
    def __init__(self, path: str):
        self.path = path
        self.fd = open(path, "wb")
        assert self.fd is not None

    def write(self, s: bytes):
        size = len(s)
        self.fd.write(
            size.to_bytes(length=8, byteorder='big', signed=False)
        )
        self.fd.write(s)

    def close(self):
        self.fd.flush()
        # print("close")

    def append(self, obj: any):
        pickle.dump(obj=obj, file=self)

    def done(self):
        self.fd.close()


if __name__ == '__main__':
    buffer = PickleWriteBuffer("./test")
    buffer.append({"PI": 3.141})
    buffer.append({"GOLD": 0.618})
    buffer.done()


class PickleReadBuffer(BytesIO):
    def __init__(self, path: str):
        self.path = path
        self.fd = open(path, "rb")
        assert self.fd is not None

    def read(self, __size: Optional[int] = ...) -> bytes:
        size = int.from_bytes(self.fd.read(8), byteorder='big', signed=False)
        data = self.fd.read(size)
        return data

    def pop(self) -> any:
        return pickle.load(self)

    def done(self):
        self.fd.close()


if __name__ == '__main__':
    buffer = PickleReadBuffer("./test")
    print(buffer.pop())
    print(buffer.pop())
    buffer.done()
