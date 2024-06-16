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
