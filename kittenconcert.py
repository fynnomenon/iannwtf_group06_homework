#!/usr/bin/env python
# coding: utf-8

from cat import Cat

# Create the litter 
cat_1 = Cat('Clawdia')
cat_2 = Cat('Jennifur')
cat_3 = Cat('Picatso')
cat_4 = Cat('Hairy Potter')
cat_5 = 'TraitorDog'

# kittens greeting eachother 
cat_1.greet(cat_3)
cat_2.greet(cat_4)

# kittens hissing at a dog 
cat_3.greet(cat_5)
