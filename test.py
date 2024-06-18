import unittest
import time
from threading import Thread
import keyboard


running = True


def process(start_arg: str):
    if start_arg == "LS":
        print(1)
    elif start_arg == "LS2":
        print(2)


def run():
    global running

    print("START")

    i = 0
    while running:
        i += 1
        time.sleep(1)

    print(i)
    print("FINISH")


class TestCases(unittest.TestCase):
    def test_ls(self):
        global running

        thread = Thread(target=run)
        thread.start()

        print("")

        while True:
            if keyboard.is_pressed("k"):
                running = False
                break

    def test_ls2(self):
        process("LS2")

import time

from threading import Thread


running = True


def run():
    global running

    print("START")

    i = 0
    while running:
        i += 1
        time.sleep(1)

    print(i)
    print("FINISH")


if __name__ == "__main__":
    thread = Thread(target=run)
    thread.start()

    print("")

    while True:
        args = input("Args: ")

        if "end" in args:
            running = False
            break


for file in dir_list:
    if re.search("..*\.docx$", file):
        ts = dt.now()
        ending = str(ts.year)[2:] + '{:02d}'.format(ts.month) + \
                 '{:02d}'.format(ts.day) + '-' + '{:02d}'.format(ts.hour) + \
                 '{:02d}'.format(ts.minute) + '{:02d}'.format(ts.second)
        print(file[:-5] + '_' + ending)

