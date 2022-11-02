#!/usr/bin/env python
# coding: utf-8

class Cat():

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'Cat(name="{self.name}")'

    def greet(self, other):
        if not isinstance(other, Cat):
            raise TypeError("Kittens don't like puppies!")
        else:
            print(f'Meow! I am {self.name}! I see you are cat too, {other.name}.')
