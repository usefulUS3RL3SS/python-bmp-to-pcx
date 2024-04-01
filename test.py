from typing import NamedTuple, BinaryIO
import struct
import math
from operator import itemgetter
from itertools import chain
import array
import os
import sys
import random

from PIL import Image

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

BMP_HEADER_STRUCT = struct.Struct("<HIHHI")
BMP_INFO_STRUCT = struct.Struct("<IiiHHIIiiII")


class Pixel(NamedTuple):
    r: int
    g: int
    b: int


def split_bucket(bucket: list[Pixel]) -> tuple[list[Pixel], list[Pixel]]:
    r_range = (
        max(bucket, key=itemgetter(0), default=Pixel(0, 0, 0)).r
        - min(bucket, key=itemgetter(0), default=Pixel(0, 0, 0)).r
    )
    g_range = (
        max(bucket, key=itemgetter(1), default=Pixel(0, 0, 0)).g
        - min(bucket, key=itemgetter(1), default=Pixel(0, 0, 0)).g
    )
    b_range = (
        max(bucket, key=itemgetter(2), default=Pixel(0, 0, 0)).b
        - min(bucket, key=itemgetter(2), default=Pixel(0, 0, 0)).b
    )
    if r_range >= g_range >= b_range:
        bucket.sort(key=itemgetter(0))
    elif g_range >= b_range >= r_range:
        bucket.sort(key=itemgetter(1))
    else:
        bucket.sort(key=itemgetter(2))
    return bucket[: len(bucket) // 2], bucket[len(bucket) // 2 :]


def median_cut(image_data: list[list[Pixel]], k: int) -> tuple[list[list[int]], list[Pixel]]:
    buckets = [list(chain.from_iterable(image_data))]
    while len(buckets) < k:
        new_buckets = []
        for bucket in buckets:
            new_buckets.extend(split_bucket(bucket))
        buckets = new_buckets
    palette = []
    pixel_indexes = {}
    for i, bucket in enumerate(buckets):
        r_med = 0
        g_med = 0
        b_med = 0
        for pixel in bucket:
            r_med += pixel.r / len(bucket)
            g_med += pixel.g / len(bucket)
            b_med += pixel.b / len(bucket)
            pixel_indexes[pixel] = i

        palette.append(Pixel(int(r_med), int(g_med), int(b_med)))
    new_image_data = [[pixel_indexes[pixel] for pixel in line] for line in image_data]
    return new_image_data, palette


class BMPHeader(NamedTuple):
    type: int
    size: int
    reserved1: int
    reserved2: int
    off_bits: int


class BMPInfo(NamedTuple):
    size: int
    width: int
    height: int
    planes: int
    bit_count: int
    compression: int
    size_image: int
    x_pels_per_meter: int
    Y_pels_per_meter: int
    clr_used: int
    clr_important: int


# BYTE — 8-битное беззнаковое целое.
# WORD — 16-битное беззнаковое целое.
# DWORD — 32-битное беззнаковое целое.
# LONG — 32-битное целое со знаком.


class BMP:
    def __init__(self, pixel_data: list[list[Pixel]]):
        self.pixel_data = pixel_data

    @staticmethod
    def _calc_line_length(width: int) -> int:
        return math.ceil(width / 4) * 4

    @staticmethod
    def _get_pixel_data(raw_data: bytes, header: BMPHeader, info: BMPInfo) -> list[list[Pixel]]:
        pixel_data = []
        # if info.bit_count == 24:
        #     line_length = BMP._calc_line_length(info.width * 3)
        #     for line in range(info.height):
        #         scan = []
        #         for col in range(info.width):
        #             offset = header.off_bits + line_length * line + col * 3
        #             scan.append(Pixel(*raw_data[offset + 2 : offset - 1 : -1]))
        #         pixel_data.append(scan)
        # elif info.bit_count == 8:
        line_length = BMP._calc_line_length(info.width)
        palette = []
        for i in range(256):
            offset = 0x36 + i * 4
            palette.append(Pixel(*raw_data[offset + 2 : offset - 1 : -1]))
        pixel_data = [
            [palette[raw_data[header.off_bits + line_length * line + col]] for col in range(info.width)]
            for line in range(info.height)
        ]

        return pixel_data

    @classmethod
    def from_file(cls, file: BinaryIO) -> "BMP":
        raw_data = file.read()
        header = BMPHeader(*BMP_HEADER_STRUCT.unpack_from(raw_data))
        info = BMPInfo(*BMP_INFO_STRUCT.unpack_from(raw_data, offset=BMP_HEADER_STRUCT.size))
        pixel_data = cls._get_pixel_data(raw_data, header, info)
        return cls(pixel_data)

    def to_file(self, file: BinaryIO, colors):
        buf: list[bytes] = []
        new_pixel_data, new_palette = median_cut(self.pixel_data, colors)
        new_header = BMPHeader(
            19778,
            BMP_HEADER_STRUCT.size
            + BMP_INFO_STRUCT.size
            + colors * 4
            + self._calc_line_length(len(self.pixel_data[0]) if colors == 256 else len(self.pixel_data[0]) // 2)
            * len(self.pixel_data),
            0,
            0,
            BMP_HEADER_STRUCT.size + BMP_INFO_STRUCT.size + colors * 4,
        )
        print(new_header)
        new_info = BMPInfo(
            40,
            len(self.pixel_data[0]),
            len(self.pixel_data),
            1,
            int(math.log2(colors)),
            0,
            0,  # Мб тут другое
            0,
            0,
            colors,
            colors,
        )
        print(new_info)
        buf.append(BMP_HEADER_STRUCT.pack(*new_header))
        buf.append(BMP_INFO_STRUCT.pack(*new_info))
        for color in new_palette:
            buf.append(struct.pack("BBBB", color.b, color.g, color.r, 0))

        for line in new_pixel_data:
            arr = array.array("B")
            arr.fromlist(line)
            padding = self._calc_line_length(len(self.pixel_data[0])) - len(self.pixel_data[0])

            buf.append(arr.tobytes())
            buf.append(array.array("B", [0] * padding).tobytes())

        sz = 0
        for i in buf:
            sz += len(i)
            file.write(i)


PCX_HEADER_STRUCT = struct.Struct("<BBBBHHHHHH")
PCX_HEADER_STRUCT2 = struct.Struct("<BBHHHH")


class PCXHeader(NamedTuple):
    type: int
    version: int
    encoding: int
    bits_per_pixel: int
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    horizontal_dpi: int
    vertical_dpi: int


class PCXHeader2(NamedTuple):
    reserved: int
    color_planes: int
    bytes_of_one_color_plane: int
    mode_to_construct_palette: int
    horizontal_resolution: int
    vertical_rezolution: int


# TRUE COLOR
# PCXHeader(type=10, version=5, encoding=0, bits_per_pixel=8, min_x=0, min_y=0, max_x=369, max_y=801, horizontal_dpi=0, vertical_dpi=0)
# PCXHeader2(reserved=0, color_planes=3, bytes_of_one_color_plane=370, mode_to_construct_palette=1, horizontal_resolution=0, vertical_rezolution=0)

# 256
# PCXHeader(type=10, version=5, encoding=0, bits_per_pixel=8, min_x=0, min_y=0, max_x=369, max_y=801, horizontal_dpi=0, vertical_dpi=0)
# PCXHeader2(reserved=0, color_planes=1, bytes_of_one_color_plane=370, mode_to_construct_palette=1, horizontal_resolution=0, vertical_rezolution=0)

# 16
# PCXHeader(type=10, version=5, encoding=0, bits_per_pixel=8, min_x=0, min_y=0, max_x=369, max_y=801, horizontal_dpi=0, vertical_dpi=0)
# PCXHeader2(reserved=0, color_planes=1, bytes_of_one_color_plane=370, mode_to_construct_palette=1, horizontal_resolution=0, vertical_rezolution=0)


class PCX:
    def __init__(self, pixel_data):
        self.pixel_data = pixel_data

    @staticmethod
    def _get_pixel_data(raw_data: bytes, header: PCXHeader, header2: PCXHeader2) -> list[list[Pixel]]:
        pixel_data = []
        if header2.color_planes == 3:  # True color
            for line in range(header.max_y + 1):
                row = []
                for col in range(header.max_x + 1):
                    row.append(
                        Pixel(
                            raw_data[128 + line * 3 * (header.max_x + 1) + col],
                            raw_data[128 + (line * 3 + 1) * (header.max_x + 1) + col],
                            raw_data[128 + (line * 3 + 2) * (header.max_x + 1) + col],
                        )
                    )
                pixel_data.append(row)
        else:
            palette = raw_data[-768:]
            for line in range(header.max_y + 1):
                row = []
                for col in range(header.max_x + 1):
                    palette_idx = raw_data[128 + line * (header.max_x + 1) + col] * 3
                    row.append(Pixel(*palette[palette_idx : palette_idx + 3]))
                pixel_data.append(row)

        return pixel_data

    @classmethod
    def from_file(cls, file: BinaryIO) -> "PCX":
        raw_data = file.read()
        header = PCXHeader(*PCX_HEADER_STRUCT.unpack_from(raw_data))
        header2 = PCXHeader2(*PCX_HEADER_STRUCT2.unpack_from(raw_data, offset=0x40))
        print(header)
        print(header2)
        pixel_data = cls._get_pixel_data(raw_data, header, header2)
        return cls(pixel_data)

    def to_file(self, file: BinaryIO, colors):
        buf: list[bytes] = []
        new_pixel_data, new_palette = median_cut(self.pixel_data, colors)
        new_header = PCXHeader(
            type=10,
            version=5,
            encoding=0,
            bits_per_pixel=8,
            min_x=0,
            min_y=0,
            max_x=len(self.pixel_data[0]) - 1,
            max_y=len(self.pixel_data) - 1,
            horizontal_dpi=0,
            vertical_dpi=0,
        )
        new_header2 = PCXHeader2(
            reserved=0,
            color_planes=1,
            bytes_of_one_color_plane=len(self.pixel_data[0]),
            mode_to_construct_palette=1,
            horizontal_resolution=0,
            vertical_rezolution=0,
        )
        print(new_header)
        print(new_header2)
        buf.append(PCX_HEADER_STRUCT.pack(*new_header))
        buf.append(array.array("B", [0] * 48).tobytes())
        buf.append(PCX_HEADER_STRUCT2.pack(*new_header2))
        buf.append(array.array("B", [0] * 54).tobytes())
        buf.append(array.array("B", chain.from_iterable(new_pixel_data)).tobytes())
        buf.append(bytes([0x0C]))
        buf.append(array.array("B", chain.from_iterable(new_palette)).tobytes())
        if colors == 16:
            buf.append(array.array("B", [0] * (256 - 16) * 3).tobytes())
        sz = 0
        for i in buf:
            sz += len(i)
            file.write(i)


bmp = BMP.from_file(open("input.bmp", "rb")) 
pcx = PCX(list(reversed(bmp.pixel_data)))
pcx.to_file(open("result.pcx", "wb"), 16)





def printPcxFile():
    class RGB:
        """Структура для хранения красного, зеленого и синего канала
        и зарезервированного поля
        """

        def __init__(self, r, g, b):
            self.r = r
            self.g = g
            self.b = b

        def __repr__(self):
            return f"R{self.r}  G{self.g}  B{self.b}"


    def read_int_from_file(binary_file, size):
        """Функция считывает из файла binary_file ровно size
        байт и возвращает образуемое ими в little-endian число
        """
        return int.from_bytes(
            binary_file.read(size),
            byteorder='little'
        )


    def read_BYTE(binary_file):
        return read_int_from_file(binary_file, 1)


    def read_WORD(binary_file):
        return read_int_from_file(binary_file, 2)


    def read_RGB(binary_file):
        return RGB(
            read_BYTE(binary_file),
            read_BYTE(binary_file),
            read_BYTE(binary_file)
        )


    class PCXIMAGE:
        def __init__(self, pcx_file):
            self.header = PCXFILEHEADER(pcx_file)
            self.width = self.header.x_max + 1
            self.height = self.header.y_max + 1

            byte_beacon_num = pcx_file.seek(-769, 2)
            byte_beacon = True if (hex(int.from_bytes(pcx_file.read(1), byteorder="little")) == hex(
                0xC) or hex(int.from_bytes(pcx_file.read(1), byteorder="little")) == hex(0xA)) else False

            if byte_beacon:
                self.colormap = list()
                for i in range(256):
                    self.colormap.append(read_RGB(pcx_file))
            else:
                self.colormap = self.header.colormap

            self.img_arr = list()
            pcx_file.seek(128, 0)
            while (byte := pcx_file.read(1)):
                if byte_beacon and not pcx_file.tell() < byte_beacon_num:
                    break

                byte = int.from_bytes(byte, byteorder='little')

                if (byte >> 6) == 0b11:
                    counter = byte & 0x3F
                    byte = pcx_file.read(1)
                    byte = int.from_bytes(byte, byteorder='little')
                    for i in range(counter):
                        self.img_arr.append(self.colormap[byte])
                    # self.img_arr = self.img_arr + ([self.colormap[byte]] * counter)
                else:
                    self.img_arr.append(self.colormap[byte])

            print(len(self.img_arr), self.width * self.height)

        def info(self):
            return self.header.info()


    class PCXFILEHEADER:
        """ Структура с информационными полями (128) """

        def __init__(self, pcx_file):
            self.manufacturer = read_BYTE(pcx_file)
            self.version = read_BYTE(pcx_file)
            self.encoding = read_BYTE(pcx_file)
            self.bits_per_pixel = read_BYTE(pcx_file)
            self.x_min = read_WORD(pcx_file)
            self.y_min = read_WORD(pcx_file)
            self.x_max = read_WORD(pcx_file)
            self.y_max = read_WORD(pcx_file)
            self.hdpi = read_WORD(pcx_file)
            self.vdpi = read_WORD(pcx_file)
            self.colormap = list()
            for i in range(16):
                self.colormap.append(read_RGB(pcx_file))
            self.reserved = read_BYTE(pcx_file)
            self.num_planes = read_BYTE(pcx_file)
            self.bytes_per_line = read_WORD(pcx_file)
            self.palette_info = read_WORD(pcx_file)
            self.h_screen_size = read_WORD(pcx_file)
            self.v_screen_size = read_WORD(pcx_file)
            self.filler = read_int_from_file(pcx_file, 54)

        def info(self):
            """ Ф-я возвращает информацию о структуре """

            info_str = f"PCXFILEHEADER\n" \
                f"  Manufacturer: {self.manufacturer}\n" \
                f"  Version: {self.version}\n" \
                f"  Encoding: {self.encoding}\n" \
                f"  BitsPerPixel: {self.bits_per_pixel}\n" \
                f"  Xmin: {self.x_min}\n" \
                f"  Ymin: {self.y_min}\n" \
                f"  Xmax: {self.x_max}\n" \
                f"  Ymax: {self.y_max}\n" \
                f"  HDpi: {self.hdpi}\n" \
                f"  VDpi: {self.vdpi}\n" \
                f"  Colormap: {[str(i) for i in self.colormap]}\n" \
                f"  Reserved: {self.reserved}\n" \
                f"  NPlanes: {self.num_planes}\n" \
                f"  BytesPerLine: {self.bytes_per_line}\n" \
                f"  PaletteInfo: {self.palette_info}\n" \
                f"  HscreenSize: {self.h_screen_size}\n" \
                f"  VscreenSize: {self.v_screen_size}\n" \
                f"  Filler: {self.filler}"
            return info_str


    class Widget(QWidget):
        def __init__(self):
            super().__init__()

            file = open("result.PCX", 'rb')
            
            self.pcx = PCXIMAGE(file)
            print(self.pcx.info())

            self.resize(self.pcx.width, self.pcx.height)
            self.setWindowTitle('Лабораторная работа #8')
            self.center()

        def paintEvent(self, event):
            super().paintEvent(event)

            painter = QPainter(self)
            painter.setBrush(Qt.black)
            painter.drawRect(self.rect())

            painter.setPen(QColor(0xff, 0xff, 0xff))
            painter.drawPoint(self.width() // 2, self.height() // 2)
            for i in range(self.pcx.width):
                for j in range(self.pcx.height):
                    color = self.pcx.img_arr[j * self.pcx.header.bytes_per_line + i]
                    painter.setPen(QColor(color.r, color.g, color.b))
                    painter.drawPoint(i, j)

        def center(self):
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())


        app = QApplication(sys.argv)
        w = Widget()
        w.show()
        sys.exit(app.exec_())